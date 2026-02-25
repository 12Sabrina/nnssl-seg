import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import time
import json
from monai.data import DataLoader
from monai.data.utils import pad_list_data_collate
from monai.optimizers import WarmupCosineSchedule

# nnSSL imports
from nnssl.architectures.architecture_registry import get_res_enc_l
from nnssl.run.load_pretrained_weights import load_pretrained_weights
from nnssl.dataset.multimodal_dataset import get_datasets, format_input
from nnssl.dataset.transforms import custom_transform
from nnssl.loss.dice import EDiceLoss
from nnssl.train_utils import cal_metrics_extended, dice_metric, post_trans, compute_hd95
from nnssl.inference_util import sliding_window_inference
from pathlib import Path

def add_seg_args(parser):
    parser.add_argument("--datalist-path", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="/")
    parser.add_argument("--arch", type=str, default="ResEncL")
    parser.add_argument("--pretrained-weights", type=str)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--epoch-length", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser

def train_iter(model, batch, optimizer, scheduler, loss_function, scaler):
    x = batch["image"].cuda()
    y = batch["label"].cuda()
    
    with torch.cuda.amp.autocast():
        logits = model(x)
        if isinstance(logits, (list, tuple)):
            loss = 0
            weights = [1 / (2**i) for i in range(len(logits))]
            weights = [w / sum(weights) for w in weights]
            for i, l in enumerate(logits):
                loss += weights[i] * loss_function(l, y)
        else:
            loss = loss_function(logits, y)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return loss.item()

def val_iter(model, batch, image_size, overlap=0.5):
    x = batch["image"].cuda()
    y = batch["label"].cuda()
    
    def model_forward(inputs, crop_indexes=None):
        out = model(inputs)
        return out[0] if isinstance(out, (list, tuple)) else out

    logits = sliding_window_inference(x, (image_size,)*3, 1, model_forward, overlap=overlap)
    preds = post_trans(logits)
    dice_metric(y_pred=preds, y=y)
    
    y_np = y.cpu().numpy()[0]
    p_np = preds.cpu().numpy()[0]
    
    hd95_list = []
    l_dice_list = []
    l_hd95_list = []
    
    for c in range(y_np.shape[0]):
        hd = compute_hd95(y_np[c], p_np[c])
        hd95_list.append(hd)
        l_dice_list.append(0.0) 
        l_hd95_list.append(0.0)
        
    return hd95_list, l_dice_list, l_hd95_list

def do_run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    print("--- Configuration ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("---------------------")

    with open(args.datalist_path, 'r') as f:
        data_list = json.load(f)
    
    from monai.data import Dataset
    data_root = args.data_root
    train_files = format_input(Path(data_root), data_list.get("training", []))
    val_files = format_input(Path(data_root), data_list.get("validation", []))
    test_files = format_input(Path(data_root), data_list.get("test", []))
    
    train_trans = custom_transform(patch_shape=args.image_size, mode='train')
    val_trans = custom_transform(patch_shape=args.image_size, mode='val')
    
    train_ds = Dataset(data=train_files, transform=train_trans)
    val_ds = Dataset(data=val_files, transform=val_trans)
    test_ds = Dataset(data=test_files, transform=val_trans)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = get_res_enc_l(num_input_channels=1, num_output_channels=3, deep_supervision=False)
    if args.pretrained_weights:
        load_pretrained_weights(model, args.pretrained_weights, verbose=True)
    
    model.cuda()
    loss_fn = EDiceLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    max_iter = args.epochs * args.epoch_length
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=0, t_total=max_iter)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(train_loader):
            if i >= args.epoch_length: break
            loss = train_iter(model, batch, optimizer, scheduler, loss_fn, scaler)
            epoch_loss += loss
            print(f"Train epoch [{epoch+1}/{args.epochs}]({i}/{args.epoch_length}):  loss: {loss:.4f}")

        # Validation
        model.eval()
        dice_metric.reset()
        hd95_metrics, l_dice_metrics, l_hd95_metrics = [], [], []
        with torch.no_grad():
            for val_data in val_loader:
                h, ld, lh = val_iter(model, val_data, args.image_size)
                hd95_metrics.append(h)
                l_dice_metrics.append(ld)
                l_hd95_metrics.append(lh)
        
        dice_results = dice_metric.aggregate().cpu().numpy()
        cal_metrics_extended([dice_results], hd95_metrics, l_dice_metrics, l_hd95_metrics, current_epoch=epoch+1, save_folder=args.output_dir, mode='validation')

        # Test
        dice_metric.reset()
        hd95_metrics, l_dice_metrics, l_hd95_metrics = [], [], []
        with torch.no_grad():
            for test_data in test_loader:
                h, ld, lh = val_iter(model, test_data, args.image_size)
                hd95_metrics.append(h)
                l_dice_metrics.append(ld)
                l_hd95_metrics.append(lh)
        
        dice_results = dice_metric.aggregate().cpu().numpy()
        cal_metrics_extended([dice_results], hd95_metrics, l_dice_metrics, l_hd95_metrics, current_epoch=epoch+1, save_folder=args.output_dir, mode='test')

if __name__ == "__main__":
    parser = add_seg_args(argparse.ArgumentParser())
    do_run(parser.parse_args())
