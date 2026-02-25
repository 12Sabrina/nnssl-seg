import os
from pathlib import Path
import csv
import yaml

import numpy as np
import torch
from nnssl.inference_util import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType
)


import random
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def init_config(args):
    args.ckpt_save_dir = Path(f"./{args.output_dir}/{args.experiment}/")
    args.ckpt_save_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_save_dir = args.ckpt_save_dir.resolve()
    config = vars(args).copy()
    config_file = args.ckpt_save_dir / (args.experiment + ".yaml")
    with open(config_file, "w") as file:
        yaml.dump(config, file)
    return args

def set_seed(args):
    torch.manual_seed(args.random_seed+args.rank)
    torch.cuda.manual_seed(args.random_seed+args.rank)
    torch.cuda.manual_seed_all(args.random_seed+args.rank) # multi-GPU and sample  different cases in each rank
    
    np.random.seed(args.random_seed+args.rank)
    random.seed(args.random_seed+args.rank)   
    os.environ['PYTHONHASHSEED'] = str(args.random_seed+args.rank)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def cal_metrics(metrics, current_epoch=0, save_folder=None, mode='validation'):
    metrics = list(zip(*metrics))
    metrics = [np.nanmean(torch.tensor(dice, dtype=float).cpu().numpy()) for dice in metrics]
    print(
            f'{mode} {current_epoch} epoch: ',
            f'AVG: {np.round(np.mean(metrics), 5)}, ',
            ', '.join([f'{idx}: {np.round(value, 5)}' for idx, value in enumerate(metrics)])
        )

    if save_folder:
        csv_file_path = f"{save_folder}/{mode}_dice.csv"
        file_exists = os.path.exists(csv_file_path)
        with open(csv_file_path, mode="a", newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                header = [f'{mode} epoch', 'AVG'] + [f'class {str(i)}' for i in range(len(metrics))]
                writer.writerow(header)

            row = [current_epoch, np.round(np.mean(metrics), 5)] + [np.round(value, 5) for value in metrics]
            writer.writerow(row)

    return metrics


def compute_surface_voxels(binary_mask: np.ndarray):
    if binary_mask.sum() == 0:
        return np.zeros((0, 3), dtype=np.int32)
    binary_mask = binary_mask.astype(bool)
    eroded = ndimage.binary_erosion(binary_mask)
    surface = binary_mask ^ eroded
    coords = np.array(np.nonzero(surface)).T  # z,y,x order
    return coords


def compute_hd95(gt_mask: np.ndarray, pred_mask: np.ndarray):
    if gt_mask.sum() == 0 and pred_mask.sum() == 0:
        return 0.0
    if gt_mask.sum() == 0 or pred_mask.sum() == 0:
        return float('nan')

    gt_surf = compute_surface_voxels(gt_mask)
    pred_surf = compute_surface_voxels(pred_mask)
    if gt_surf.shape[0] == 0 or pred_surf.shape[0] == 0:
        return float('nan')

    tree_pred = cKDTree(pred_surf)
    dists_gt_to_pred, _ = tree_pred.query(gt_surf, k=1)
    tree_gt = cKDTree(gt_surf)
    dists_pred_to_gt, _ = tree_gt.query(pred_surf, k=1)

    hd95 = max(np.percentile(dists_gt_to_pred, 95), np.percentile(dists_pred_to_gt, 95))
    return float(hd95)


def lesion_wise_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray):
    labels, num = ndimage.label(gt_mask)
    dice_list = []
    hd95_list = []
    for lbl in range(1, num + 1):
        lesion = (labels == lbl).astype(np.uint8)
        pred_overlap = (pred_mask > 0).astype(np.uint8)
        inter = (lesion & pred_overlap).sum()
        denom = lesion.sum() + pred_overlap.sum()
        if denom == 0:
            dice = float('nan')
        else:
            dice = 2.0 * inter / denom
        dice_list.append(dice)
        hd = compute_hd95(lesion, pred_overlap)
        hd95_list.append(hd)
    return dice_list, hd95_list


def cal_metrics_extended(dice_metrics, hd95_metrics, lesionwise_dice, lesionwise_hd95, current_epoch=0, save_folder=None, mode='validation'):
    class_dices = cal_metrics(dice_metrics, current_epoch=current_epoch, save_folder=save_folder, mode=mode)
    num_classes = len(class_dices)

    hd95_arr = []
    for v in hd95_metrics:
        for x in v:
            if not np.isnan(x):
                hd95_arr.append(x)
    avg_hd95 = np.nan if len(hd95_arr) == 0 else float(np.mean(hd95_arr))

    # lesionwise_dice is structured as: list_over_batches -> list_over_classes -> list_of_lesion_metrics
    # flatten safely and ignore NaNs (some entries may be empty lists or arrays)
    lesion_dice_vals = []
    lesion_dice_by_class = [[] for _ in range(num_classes)]
    for per_case in lesionwise_dice:
        for class_idx, class_list in enumerate(per_case):
            arr = np.asarray(class_list)
            if arr.size == 0:
                continue
            for v in arr.ravel():
                if not np.isnan(v):
                    lesion_dice_vals.append(float(v))
                    lesion_dice_by_class[class_idx].append(float(v))

    lesion_hd95_vals = []
    lesion_hd95_by_class = [[] for _ in range(num_classes)]
    for per_case in lesionwise_hd95:
        for class_idx, class_list in enumerate(per_case):
            arr = np.asarray(class_list)
            if arr.size == 0:
                continue
            for v in arr.ravel():
                if not np.isnan(v):
                    lesion_hd95_vals.append(float(v))
                    lesion_hd95_by_class[class_idx].append(float(v))

    avg_lesion_dice = float(np.mean(lesion_dice_vals)) if len(lesion_dice_vals) > 0 else float('nan')
    avg_lesion_hd95 = float(np.mean(lesion_hd95_vals)) if len(lesion_hd95_vals) > 0 else float('nan')
    lesion_dice_per_class = [float(np.mean(vals)) if len(vals) > 0 else float('nan') for vals in lesion_dice_by_class]
    lesion_hd95_per_class = [float(np.mean(vals)) if len(vals) > 0 else float('nan') for vals in lesion_hd95_by_class]

    print(f"{mode} {current_epoch} epoch summary: AVG Dice: {np.round(np.mean(class_dices),5)}, AVG HD95: {np.round(avg_hd95,5) if not np.isnan(avg_hd95) else 'nan'}")
    print(f"{mode} {current_epoch} epoch summary (Lesion-wise): Dice: {np.round(avg_lesion_dice,5) if not np.isnan(avg_lesion_dice) else 'nan'}, HD95: {np.round(avg_lesion_hd95,5) if not np.isnan(avg_lesion_hd95) else 'nan'}")
    print(f"{mode} {current_epoch} epoch summary (Lesion-wise per class): Dice: {np.round(lesion_dice_per_class,5)}, HD95: {np.round(lesion_hd95_per_class,5)}")

    if save_folder:
        import csv, os
        hd95_csv = f"{save_folder}/{mode}_hd95.csv"
        write_header = not os.path.exists(hd95_csv) or os.path.getsize(hd95_csv) == 0
        with open(hd95_csv, mode="a", newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([f"{mode} epoch", "AVG_HD95"])
            writer.writerow([current_epoch, np.round(avg_hd95,5) if not np.isnan(avg_hd95) else 'nan'])

        lesion_csv = f"{save_folder}/{mode}_lesionwise.csv"
        write_header = not os.path.exists(lesion_csv) or os.path.getsize(lesion_csv) == 0
        with open(lesion_csv, mode="a", newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([f"{mode} epoch", "lesion_dice_mean", "lesion_hd95_mean"])
            writer.writerow([current_epoch, np.round(avg_lesion_dice,5) if not np.isnan(avg_lesion_dice) else 'nan', np.round(avg_lesion_hd95,5) if not np.isnan(avg_lesion_hd95) else 'nan'])

            return class_dices, avg_hd95, avg_lesion_dice, avg_lesion_hd95, lesion_dice_per_class, lesion_hd95_per_class

def save_checkpoint(state: dict, save_folder: Path):
    best_filename = str(save_folder) + '/model_best' + "_" + str(state["epoch"]) + '.pth.tar'
    torch.save(state, best_filename)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def inference(input, model, patch_shape = 128):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(patch_shape, patch_shape, patch_shape),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5
        )
    with torch.cuda.amp.autocast():
        return _compute(input)

post_trans = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)

dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")