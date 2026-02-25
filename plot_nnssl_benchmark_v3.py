import matplotlib.pyplot as plt
import os
import re
import numpy as np

def extract_test_dice(log_path):
    if not os.path.exists(log_path):
        return None
    
    dices = []
    # Pattern to match: "test 1 epoch summary: AVG Dice: 0.15021"
    pattern = re.compile(r"test \d+ epoch summary: AVG Dice: ([\d.]+)")
    
    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                dices.append(float(match.group(1)))
    return dices if dices else None

def plot_fold(real_vals, mixed_vals, fold, ax=None):
    if ax is None:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
    
    if real_vals is not None:
        ax.plot(real_vals, label='Real', color='blue', linewidth=2)
    if mixed_vals is not None:
        ax.plot(mixed_vals, label='Mixed', color='red', linewidth=2)
        
    ax.set_title(f'Fold {fold}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Avg Dice')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

def main():
    nnssl_dir = "/gpfs/share/home/2401111663/syy/nnssl-openneuro/sbatch/output/ped_v3"
    output_dir = "/gpfs/share/home/2401111663/syy/nnssl-openneuro/plots_benchmark_v3_style"
    os.makedirs(output_dir, exist_ok=True)
    
    folds = [1, 2, 3, 4, 5]
    per_fold_data = {}
    
    for f in folds:
        real_log = os.path.join(nnssl_dir, f"real_fold{f}.log")
        mixed_log = os.path.join(nnssl_dir, f"mixed_fold{f}.log")
        
        real_dice = extract_test_dice(real_log)
        mixed_dice = extract_test_dice(mixed_log)
        per_fold_data[f] = (real_dice, mixed_dice)

    # Combined figure with subplots
    n = len(folds)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
        
    for ax, f in zip(axes, folds):
        real, mixed = per_fold_data[f]
        if real is None and mixed is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'Fold {f}')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        plot_fold(real, mixed, f, ax=ax)

    plt.tight_layout()
    out_all = os.path.join(output_dir, 'nnssl_dice_comparison_all_folds.png')
    fig.savefig(out_all, dpi=300)
    print(f'Combined plot saved as {out_all}')

    # Also save per-fold images
    for f in folds:
        real, mixed = per_fold_data[f]
        if real is None and mixed is None:
            continue
        plt.figure(figsize=(8, 5))
        plot_fold(real, mixed, f)
        out_name = os.path.join(output_dir, f'nnssl_dice_comparison_fold{f}.png')
        plt.tight_layout()
        plt.savefig(out_name, dpi=300)
        plt.close() # Close to avoid memory issues with many folds
        print(f'Saved {out_name}')

if __name__ == '__main__':
    main()
