import monai.transforms as transforms
from monai.transforms import CenterSpatialCropd, EnsureTyped
from monai.transforms.utility.dictionary import ConvertToMultiChannelBasedOnBratsClassesD
from .mix_transform import RandomChannelCutmixd
import numpy as np


def _clone_tensor(x):
    tensor = x.as_tensor() if hasattr(x, 'as_tensor') else x
    return tensor.clone()


def _normalize_et_label(label):
    arr = label.clone() if hasattr(label, 'clone') else np.copy(label)
    arr[arr == 3] = 4
    return arr


def custom_transform(patch_shape=96, mode='train',enable_channel_cutmix=False, pair_aug=False, in_channels=4):
    if mode == 'train':
        if enable_channel_cutmix:
            train_transform = transforms.Compose([
                transforms.LoadImaged(keys=['image', 'label', 'template']),
                transforms.EnsureChannelFirstd(keys=['image', 'label', 'template']),
                transforms.ScaleIntensityRangePercentilesd(keys=['image'], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True, channel_wise=True),
                transforms.Orientationd(keys=['image', 'label', 'template'], axcodes='RAS'),
                transforms.Spacingd(keys=['image', 'label', 'template'], pixdim=(1.0, 1.0, 1.0), mode=['bilinear', 'nearest', 'bilinear']),
                transforms.CropForegroundd(keys=['image', 'template', 'label'], source_key='image', margin=1),
                RandomChannelCutmixd(keys=['image'], num_mix=3, pair_aug=pair_aug, max_num_channel=in_channels),
                transforms.RandSpatialCropd(keys=['image', 'label'], roi_size=patch_shape, random_size=False),  
                transforms.SpatialPadd(keys=['image', 'label'], spatial_size=patch_shape, mode='constant'),
                transforms.RandFlipd(keys=['image', 'label'], spatial_axis=(0, 1, 2), prob=0.5),
                transforms.RandShiftIntensityd(keys=['image'], offsets=0.1, prob=1.0),
                transforms.RandScaleIntensityd(keys=['image'], factors=0.1, prob=1.0),
                transforms.Lambdad(keys=['label'], func=_normalize_et_label, track_meta=False),
                ConvertToMultiChannelBasedOnBratsClassesD(keys='label'), # region-based segmentation   
            ])
        else:
            train_transform = transforms.Compose([
                transforms.LoadImaged(keys=['image', 'label']),
                transforms.EnsureChannelFirstd(keys=['image', 'label']),
                transforms.ScaleIntensityRangePercentilesd(keys=['image'], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True, channel_wise=True),
                transforms.Orientationd(keys=['image', 'label'], axcodes='RAS'),
                transforms.Spacingd(keys=['image', 'label'], pixdim=(1.0, 1.0, 1.0), mode=['bilinear', 'nearest']),
                transforms.CropForegroundd(keys=['image', 'label'], source_key='image', margin=1),
                transforms.RandSpatialCropSamplesd(keys=['image', 'label'], num_samples=4, roi_size=(patch_shape,)*3, random_size=False),
                transforms.SpatialPadd(keys=['image', 'label'], spatial_size=(patch_shape,)*3, mode='constant'),
                transforms.RandFlipd(keys=['image', 'label'], spatial_axis=(0, 1, 2), prob=0.5),
                transforms.RandShiftIntensityd(keys=['image'], offsets=0.1, prob=1.0),
                transforms.RandScaleIntensityd(keys=['image'], factors=0.1, prob=1.0),
                transforms.Lambdad(keys=['label'], func=_normalize_et_label, track_meta=False),
                ConvertToMultiChannelBasedOnBratsClassesD(keys='label'), # region-based segmentation
            ])
        return train_transform
    
    val_transform = transforms.Compose([
            transforms.LoadImaged(keys=['image', 'label']),
            transforms.EnsureChannelFirstd(keys=['image', 'label']),
            transforms.ScaleIntensityRangePercentilesd(keys=['image'], lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True, channel_wise=True),
            transforms.Orientationd(keys=['image', 'label'], axcodes='RAS'),
            transforms.Spacingd(keys=['image', 'label'], pixdim=(1.0, 1.0, 1.0), mode=['bilinear', 'nearest']),
            transforms.CropForegroundd(keys=['image', 'label'], source_key='image', margin=1),
            transforms.Lambdad(keys=['label'], func=_normalize_et_label, track_meta=False),
            ConvertToMultiChannelBasedOnBratsClassesD(keys='label'), # region-based segmentation
        ])
    return val_transform


def classification_transform(patch_shape=96, mode='train'):
    common = [
        transforms.LoadImaged(keys=['image']),
        transforms.EnsureChannelFirstd(keys=['image']),
        transforms.ScaleIntensityRangePercentilesd(keys=['image'], lower=5, upper=95, b_min=0.0, b_max=1.0, channel_wise=True),
        transforms.Orientationd(keys=['image'], axcodes='RAS'),
        transforms.SpatialPadd(keys=['image'], spatial_size=patch_shape, mode='constant'),
        EnsureTyped(keys=['image'], track_meta=False),
        transforms.Lambdad(keys=['image'], func=_clone_tensor, track_meta=False),
    ]
    if mode == 'train':
        return transforms.Compose([
            *common[:-1],
            transforms.Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear'),
            transforms.CropForegroundd(keys=['image'], source_key='image', margin=1),
            transforms.RandSpatialCropd(keys=['image'], roi_size=patch_shape, random_size=False),
            transforms.RandFlipd(keys=['image'], spatial_axis=(0, 1, 2), prob=0.5),
            transforms.RandShiftIntensityd(keys=['image'], offsets=0.1, prob=1.0),
            transforms.RandScaleIntensityd(keys=['image'], factors=0.1, prob=1.0),
            EnsureTyped(keys=['image'], track_meta=False),
            transforms.Lambdad(keys=['image'], func=_clone_tensor, track_meta=False),
        ])
    return transforms.Compose([
        *common[:-1],
        transforms.Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear'),
        transforms.CropForegroundd(keys=['image'], source_key='image', margin=1),
        CenterSpatialCropd(keys=['image'], roi_size=patch_shape),
        EnsureTyped(keys=['image'], track_meta=False),
        transforms.Lambdad(keys=['image'], func=_clone_tensor, track_meta=False),
    ])
