from pathlib import Path
import os
import json
from monai.data import Dataset
from .transforms import custom_transform

def get_modality_names(dataset_name: str = 'upenngbm') -> list:
    if dataset_name == 'isles22':
        modality_list = ['ADC', 'DWI', 'FLAIR']
    elif dataset_name == 'mrbrains13':
        modality_list = ['T1', 'T1', 'T2']
    elif dataset_name == 'vsseg':
        modality_list = ['T2']
    else:
        modality_list = ['T1', 'T1C', 'T2', 'FLAIR']
    return modality_list

def format_input(data_root: Path, 
                 filename_list: list, 
                 mix_template: bool = False, 
                 template_dir: str = '',
                 dataset_name: str = '') -> list:
    fullname_list = []
    for x in filename_list:
        fullname_list.append({
            'image': [str(data_root / y) for y in x['image']],
            'label': str(data_root / x['label'])
        })
    
    if mix_template:
        templates = {'template': [str(Path(template_dir)/ (y+'.nii.gz')) for y in get_modality_names(dataset_name)]}
        fullname_list = [{**x, **templates} for x in fullname_list]
    return fullname_list

def get_datasets(dataset, data_root, json_file, patch_shape, mix_template=False, template_dir='', use_cl=False):
    data_root = Path(data_root)
    
    if os.path.isabs(json_file):
        json_path = json_file
    else:
        json_path = os.path.join(data_root, json_file)

    with open(json_path, 'r') as fr:
        data_list = json.load(fr)

    train_list = format_input(data_root, data_list.get("training", []), mix_template, template_dir, dataset)
    val_list = format_input(data_root, data_list.get("validation", []))
    test_list = format_input(data_root, data_list.get("test", []))
    
    print(f"Train: {len(train_list)} \nValidation: {len(val_list)} \nTest: {len(test_list)}")
    
    train_transform = custom_transform(patch_shape=patch_shape, mode='train', enable_channel_cutmix=mix_template, pair_aug=use_cl)
    val_transform = custom_transform(patch_shape=patch_shape, mode='val')
    
    train_dataset = Dataset(data=train_list, transform=train_transform)
    val_dataset = Dataset(data=val_list, transform=val_transform)
    
    return train_dataset, val_dataset
