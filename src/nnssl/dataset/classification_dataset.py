import json
from pathlib import Path
from typing import List

import torch
from monai.data import Dataset

from .transforms import classification_transform

_MODALITY_KEYS = [
    'T1-weighted',
    'T1-weighted Contrast Enhanced',
    'T2-weighted',
    'T2-weighted FLAIR',
]

_PATH_REMAPS = [
    ('/gpfs/share/home/2301210592', '/gpfs/share/home/2401111663'),
]


def _resolve_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    try:
        if candidate.exists():
            return candidate
    except PermissionError:
        pass
    for src, dst in _PATH_REMAPS:
        if raw_path.startswith(src):
            remapped = Path(dst + raw_path[len(src):])
            if remapped.exists():
                return remapped
    raise FileNotFoundError(f"file not found or inaccessible: {raw_path}")


def _parse_entry(entry: dict) -> dict:
    modalities = entry.get('modalities', {}) or {}
    image_paths = []
    for key in _MODALITY_KEYS:
        if key in modalities:
            image_paths.append(_resolve_path(modalities[key]))
    if not image_paths:
        raise ValueError(f"entry {entry.get('identity')} has no modalities")

    survival_days = entry.get('Survival_days')
    survival_label = 1 if isinstance(survival_days, (int, float)) and survival_days >= 365 else 0

    subtype = str(entry.get('subtypes') or '').lower()
    subtype_label = 1 if 'high-grade' in subtype else 0

    return {
        'image': image_paths,
        'survival_label': torch.tensor(survival_label, dtype=torch.float32),
        'subtype_label': torch.tensor(subtype_label, dtype=torch.float32),
    }


def _load_split(json_path: Path) -> List[dict]:
    with json_path.open() as fp:
        content = json.load(fp)
    records = content.get('data', [])
    if not records:
        raise ValueError(f"no data found in {json_path}")
    return [_parse_entry(record) for record in records]


def get_classification_datasets(train_json: Path, val_json: Path, test_json: Path, patch_shape: int):
    train_samples = _load_split(train_json)
    val_samples = _load_split(val_json)
    test_samples = _load_split(test_json)

    train_dataset = Dataset(data=train_samples, transform=classification_transform(patch_shape=patch_shape, mode='train'))
    val_dataset = Dataset(data=val_samples, transform=classification_transform(patch_shape=patch_shape, mode='val'))
    test_dataset = Dataset(data=test_samples, transform=classification_transform(patch_shape=patch_shape, mode='val'))

    return train_dataset, val_dataset, test_dataset
