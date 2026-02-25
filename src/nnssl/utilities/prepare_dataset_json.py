import json
import argparse
from pathlib import Path

def process_entry(entry):
    if "modalities" in entry:
        m = entry["modalities"]
        # Standard BraTS modality order
        images = [
            m.get("T1-weighted", ""),
            m.get("T1-weighted Contrast Enhanced", ""),
            m.get("T2-weighted", ""),
            m.get("T2-weighted FLAIR", "")
        ]
        # Filter out empty paths
        images = [i for i in images if i]
        label = m.get("Segmentation", "")
        return {"image": images, "label": label}
    return entry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    def load_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    train_data = load_json(args.train_json)
    val_data = load_json(args.val_json)

    train_list = train_data.get("training", train_data.get("data", []))
    val_list = val_data.get("training", val_data.get("data", []))

    processed_train = [process_entry(e) for e in train_list]
    processed_val = [process_entry(e) for e in val_list]

    output = {
        "training": processed_train,
        "validation": processed_val,
        "test": []
    }

    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=4)
    
    print(f"Dataset JSON saved to {args.output_json}")

if __name__ == "__main__":
    main()
