import json
import argparse
import random
import os
from pathlib import Path

def format_to_nnssl(entry):
    """ Converts a BraTS entry with 'modalities' into nnssl {'image': [..], 'label': ..} format """
    if "modalities" in entry:
        m = entry["modalities"]
        # Use T1c for consistency with V3 experiments
        t1c = m.get("T1-weighted Contrast CE") or m.get("T1-weighted Contrast Enhanced")
        seg = m.get("Segmentation")
        if t1c and seg:
            return {"image": [t1c], "label": seg}
    return entry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-in", type=str, required=True, help="Input V3 JSON (real or mixed)")
    parser.add_argument("--val-json", type=str, required=True, help="Original validation JSON for this fold (Test set)")
    parser.add_argument("--output-json", type=str, required=True, help="Output JSON for nnssl")
    parser.add_argument("--val-count", type=int, default=10, help="Fixed number of real samples for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    def load_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    # Load input data from V3 (might already be split, so we merge training/validation to re-split consistently if needed)
    data_in = load_json(args.json_in)
    
    # To be consistent with 3DINO split, we collect ALL samples that were originally in 'train.json'.
    raw_training_part = data_in.get("training", [])
    raw_validation_part = data_in.get("validation", [])
    raw_all = raw_training_part + raw_validation_part
    
    # Separate real and synthetic (generated) samples
    real_samples = []
    synthetic_samples = []
    
    for item in raw_all:
        is_synthetic = False
        # Check image paths for synthetic marker
        for img_path in item.get("image", []):
            if "_generated.nii.gz" in img_path:
                is_synthetic = True
                break
        
        if is_synthetic:
            synthetic_samples.append(item)
        else:
            real_samples.append(item)

    # Load original validation data (the 'val.json' for the fold), which we use as Test set
    orig_val_data = load_json(args.val_json)
    # Original val.json uses BraTS dictionary format, need to convert
    orig_test_raw = orig_val_data.get("training", orig_val_data.get("data", []))
    raw_test = [format_to_nnssl(e) for e in orig_test_raw]

    # Shuffle real samples to pick validation set
    random.shuffle(real_samples)
    
    if len(real_samples) <= args.val_count:
        val_list = real_samples
        train_real = []
        print(f"Warning: Only {len(real_samples)} real samples available. Using all for validation.")
    else:
        val_list = real_samples[:args.val_count]
        train_real = real_samples[args.val_count:]
    
    # Construct final training list: remaining real + all synthetic
    train_list = train_real + synthetic_samples
    
    output = {
        "training": train_list,
        "validation": val_list,
        "test": raw_test
    }

    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=4)
    
    print(f"Created nnssl JSON: {args.output_json}")
    print(f"  Real samples total: {len(real_samples)}")
    print(f"  Used for Validation: {len(val_list)}")
    print(f"  Used for Training:   {len(train_real)}")
    print(f"  Synthetic (Mixed) Training: {len(synthetic_samples)}")
    print(f"  Test samples:        {len(raw_test)}")

if __name__ == "__main__":
    main()
