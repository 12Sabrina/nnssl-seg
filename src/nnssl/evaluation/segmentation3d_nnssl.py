import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import time
from monai.data import DataLoader
from monai.data.utils import pad_list_data_collate
from monai.optimizers import WarmupCosineSchedule

# nnSSL imports
from nnssl.architectures.architecture_registry import get_res_enc_l
from nnssl.run.load_pretrained_weights import load_pretrained_weights
from nnssl.dataset.multimodal_dataset import get_datasets
from nnssl.dataset.transforms import *
from nnssl.loss.dice import EDiceLoss
from nnssl.train_utils import cal_metrics_extended, dice_metric, post_trans, lesion_wise_metrics, compute_hd95
from nnssl.inference_util import sliding_window_inference

def add_seg_args(parser):
    parser.add_argument("--dataset-name", type=str, required=True, choices=['brats18', 'brats20', 'brats23-ped', 'brats23-met', 'isles22','mrbrains13', 'vsseg', 'upenngbm', 'BraTS'], help="Name of finetuning dataset")
    parser.add_argument("--task-type", type=str, default="segmentation", help="Task type (ignored, for compatibility)")
    parser.add_argument("--datalist-path", type=str, help="Path to the datalist JSON file")
    parser.add_argument("--data-root", type=str, help="Base data directory")
    parser.add_argument("--arch", type=str, default="ResEncL", choices=["ResEncL", "voco"], help="nnSSL Architecture")
    parser.add_argument("--pretrained-weights", type=str, help="Pretrained model weights")
    parser.add_argument("--train-feature-model", action="store_true", help="Freeze feature model or not")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--epoch-length", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--warmup-iters", type=int, default=0)
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
    
    # Process metrics using monai metrics
    logits_processed = post_trans(logits)
    dice_metric(y_pred=logits_processed, y=y)
    
    # Calculate HD95 and Lesion-wise metrics
    y_np = y.cpu().numpy()[0] # [C, D, H, W]
    pred_np = logits_processed.cpu().numpy()[0] # [C, D, H, W]
    
    hd95_list = []
    l_dice_list = []
    l_hd95_list = []
    
    for c in range(y_np.shape[0]):
        hd = compute_hd95(y_np[c], pred_np[c])
        hd95_list.append(hd)
        
        l_dice, l_hd = lesion_wise_metrics(y_np[c], pred_np[c])
        l_dice_list.append(l_dice)
        l_hd95_list.append(l_hd)
        
    return hd95_list, l_dice_list, l_hd95_list

def do_finetune(args):
    ds_name = args.dataset_name.lower()
    if ds_name == "brats": ds_name = "upenngbm"

    print(f"Loading dataset: {ds_name}")
    train_ds, val_ds = get_datasets(
        dataset=ds_name,
        data_root=args.data_root if args.data_root else "./",
        json_file=args.datalist_path,
        patch_shape=args.image_size,
        mix_template=False
    )
    
    input_channels = 4
    num_classes = 3

    train_loader = DataLoader(dataset=train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=pad_list_data_collate)
    val_loader = DataLoader(dataset=val_ds, batch_size=1, num_workers=args.num_workers, shuffle=False)

    print(f"Building model architecture: {args.arch}")
    model = get_res_enc_l(num_input_channels=input_channels, num_output_channels=num_classes, deep_supervision=False)

    if args.pretrained_weights:
        print(f"Loading pretrained weights from {args.pretrained_weights}")
        load_pretrained_weights(model, args.pretrained_weights, verbose=True)

    if not args.train_feature_model:
        if hasattr(model, "encoder"):
            print("Freezing encoder weights.")
            for param in model.encoder.parameters():
                param.requires_grad = False
            model.encoder.eval()

    model.cuda()
    loss_fn = EDiceLoss().cuda()
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    max_iter = args.epochs * args.epoch_length
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_iters, t_total=max_iter)

    best_val_score = -1
    os.makedirs(args.output_dir, exist_ok=True)

    print("Starting training loop...")
    for epoch in range(args.epochs):
        model.train()
        if not args.train_feature_model and hasattr(model, "encoder"):
            model.encoder.eval()
        
        epoch_loss = 0
        for i, train_data in enumerate(train_loader):
            if i >= args.epoch_length: break
            loss = train_iter(model, train_data, optimizer, scheduler, loss_fn, scaler)
            epoch_loss += loss
            if (i+1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Iter [{i+1}/{args.epoch_length}] Loss: {loss:.4f}", flush=True)

        # Validation every epoch
        print(f"Evaluating epoch {epoch+1}...")
        model.eval()
        dice_metric.reset()
        hd95_metrics, l_dice_metrics, l_hd95_metrics = [], [], []
        
        with torch.no_grad():
            for v_it, val_data in enumerate(val_loader):
                h, ld, lh = val_iter(model, val_data, args.image_size)
                hd95_metrics.append(h)
                l_dice_metrics.append(ld)
                l_hd95_metrics.append(lh)
                if (v_it + 1) % 20 == 0:
                    print(f"Val Iter [{v_it+1}/{len(val_loader)}]", flush=True)

        # Aggregate and print extended metrics (Dice, HD95, Lesion-wise)
        dice_results = dice_metric.aggregate().cpu().numpy()
        cal_metrics_extended(
            [dice_results], 
            hd95_metrics, 
            l_dice_metrics, 
            l_hd95_metrics, 
            current_epoch=epoch+1, 
            save_folder=args.output_dir, 
            mode='validation'
        )
        
        avg_dice = np.mean(dice_results)
        if avg_dice > best_val_score:
            best_val_score = avg_dice
            print(f"New best Dice: {best_val_score:.5f}. Saving model...")
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

if __name__ == "__main__":
    parser = add_seg_args(argparse.ArgumentParser())
    args = parser.parse_args()
    do_finetune(args)
