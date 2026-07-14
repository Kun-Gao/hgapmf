from .augmentation import (
    AugmentationConfig,
    apply_intensity_augmentations,
    augment_ct_mr_3d,
    random_flip_3d,
    random_gamma,
    random_gaussian_noise,
    random_intensity_scale_shift,
)
from .model import (
    Config,
    HGAPMFBlock,
    HGAPMFNet,
    OUTPUT_CHANNELS,
    PROGRESSIVE_LABEL_MAPS,
    PROGRESSIVE_LEVELS,
    validate_progressive_label_maps,
)
from .preprocessing import (
    center_crop_3d,
    foreground_random_crop_3d,
    normalize_ct,
    normalize_ct_nnunet,
    normalize_mri,
    normalize_zscore_nnunet,
    pad_to_patch_size,
    random_crop_3d,
    stack_ct_mr,
)
from .spatial_refinement_cue import SpatialRefinementCuePredictor

# Backward-compatible aliases for earlier EXP-018 naming.
EXP018Config = Config
EXP018_OUTPUT_CHANNELS = OUTPUT_CHANNELS
EXP018_PROGRESSIVE_LABEL_MAPS = PROGRESSIVE_LABEL_MAPS
EXP018_PROGRESSIVE_LEVELS = PROGRESSIVE_LEVELS
HGAPMFBlockEXP018 = HGAPMFBlock
HGAPMF_EXP018_Net = HGAPMFNet

__all__ = [
    "Config",
    "OUTPUT_CHANNELS",
    "PROGRESSIVE_LABEL_MAPS",
    "PROGRESSIVE_LEVELS",
    "HGAPMFBlock",
    "HGAPMFNet",
    "SpatialRefinementCuePredictor",
    "validate_progressive_label_maps",
    "normalize_ct",
    "normalize_mri",
    "normalize_ct_nnunet",
    "normalize_zscore_nnunet",
    "stack_ct_mr",
    "pad_to_patch_size",
    "random_crop_3d",
    "foreground_random_crop_3d",
    "center_crop_3d",
    "AugmentationConfig",
    "random_flip_3d",
    "random_intensity_scale_shift",
    "random_gaussian_noise",
    "random_gamma",
    "apply_intensity_augmentations",
    "augment_ct_mr_3d",
    "EXP018Config",
    "EXP018_OUTPUT_CHANNELS",
    "EXP018_PROGRESSIVE_LABEL_MAPS",
    "EXP018_PROGRESSIVE_LEVELS",
    "HGAPMFBlockEXP018",
    "HGAPMF_EXP018_Net",
]
