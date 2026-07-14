from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def normalize_ct(
    image: np.ndarray,
    ct_clip: Sequence[float] = (-1000.0, 2000.0),
    method: str = "zscore",
    eps: float = 1e-8,
) -> np.ndarray:
    """Clip and normalize a CT volume.

    Args:
        image: Input CT volume.
        ct_clip: Lower and upper HU clipping bounds.
        method: ``"zscore"`` or ``"minmax"``.
        eps: Numerical stability constant.
    """
    image = np.clip(image.astype(np.float32), ct_clip[0], ct_clip[1])
    if method == "zscore":
        return ((image - image.mean()) / (image.std() + eps)).astype(np.float32)
    if method == "minmax":
        image = (image - ct_clip[0]) / (ct_clip[1] - ct_clip[0] + eps)
        image = image * 2.0 - 1.0
        return image.astype(np.float32)
    raise ValueError('CT normalization method must be "zscore" or "minmax".')


def normalize_mri(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Foreground z-score normalize an MR volume using non-zero voxels."""
    image = image.astype(np.float32)
    mask = image != 0
    if np.any(mask):
        mean = image[mask].mean()
        std = image[mask].std()
        out = image.copy()
        out[mask] = (out[mask] - mean) / (std + eps)
        out[~mask] = 0.0
        return out.astype(np.float32)
    return image


def normalize_ct_nnunet(image: np.ndarray, intensity_properties: dict, eps: float = 1e-8) -> np.ndarray:
    """nnU-Net-style CT normalization from dataset intensity properties."""
    image = image.astype(np.float32)
    lower = float(intensity_properties["percentile_00_5"])
    upper = float(intensity_properties["percentile_99_5"])
    mean = float(intensity_properties["mean"])
    std = max(float(intensity_properties["std"]), eps)
    image = np.clip(image, lower, upper)
    image -= mean
    image /= std
    return image.astype(np.float32)


def normalize_zscore_nnunet(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Global z-score normalization matching the simple nnU-Net z-score path."""
    image = image.astype(np.float32)
    mean = float(image.mean())
    std = max(float(image.std()), eps)
    image -= mean
    image /= std
    return image.astype(np.float32)


def stack_ct_mr(ct: np.ndarray, mri: np.ndarray) -> np.ndarray:
    """Return a two-channel ``[2, D, H, W]`` CT/MR tensor-ready array."""
    if ct.shape != mri.shape:
        raise ValueError(f"CT/MR shape mismatch: {ct.shape} vs {mri.shape}.")
    return np.stack([ct.astype(np.float32), mri.astype(np.float32)], axis=0)


def pad_to_patch_size(
    ct: np.ndarray,
    mri: np.ndarray,
    label: np.ndarray | None,
    patch_size: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Pad CT/MR and an optional label to at least ``patch_size``."""
    patch_size_arr = np.asarray(patch_size, dtype=np.int64)
    spatial_shape = np.asarray(ct.shape, dtype=np.int64)
    if mri.shape != ct.shape:
        raise ValueError(f"CT/MR shape mismatch: {ct.shape} vs {mri.shape}.")
    if label is not None and label.shape != ct.shape:
        raise ValueError(f"Image/label shape mismatch: image={ct.shape}, label={label.shape}.")

    pad_needed = np.maximum(patch_size_arr - spatial_shape, 0)
    pad_width = [(int(p // 2), int(p - p // 2)) for p in pad_needed]
    if not any(pad_needed):
        return ct, mri, label

    ct = np.pad(ct, pad_width, mode="constant", constant_values=0)
    mri = np.pad(mri, pad_width, mode="constant", constant_values=0)
    if label is not None:
        label = np.pad(label, pad_width, mode="constant", constant_values=0)
    return ct, mri, label


def _crop_slices(start: np.ndarray, patch_size: np.ndarray) -> tuple[slice, ...]:
    return tuple(slice(int(s), int(s + p)) for s, p in zip(start, patch_size))


def random_crop_3d(
    ct: np.ndarray,
    mri: np.ndarray,
    label: np.ndarray | None,
    patch_size: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Randomly crop a 3D CT/MR pair and optional label."""
    ct, mri, label = pad_to_patch_size(ct, mri, label, patch_size)
    patch_size_arr = np.asarray(patch_size, dtype=np.int64)
    shape = np.asarray(ct.shape, dtype=np.int64)
    max_start = shape - patch_size_arr
    start = np.asarray(
        [np.random.randint(0, int(m + 1)) if m > 0 else 0 for m in max_start],
        dtype=np.int64,
    )
    slices = _crop_slices(start, patch_size_arr)
    return ct[slices], mri[slices], None if label is None else label[slices]


def foreground_random_crop_3d(
    ct: np.ndarray,
    mri: np.ndarray,
    label: np.ndarray,
    patch_size: Sequence[int],
    foreground_classes: Sequence[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random crop centered near foreground labels when foreground exists."""
    ct, mri, padded_label = pad_to_patch_size(ct, mri, label, patch_size)
    if padded_label is None:
        raise ValueError("foreground_random_crop_3d requires a label array.")
    patch_size_arr = np.asarray(patch_size, dtype=np.int64)
    shape = np.asarray(padded_label.shape, dtype=np.int64)

    if foreground_classes is None:
        mask = padded_label > 0
    else:
        mask = np.isin(padded_label, np.asarray(foreground_classes, dtype=padded_label.dtype))

    coords = np.argwhere(mask)
    if coords.size == 0:
        cropped_ct, cropped_mri, cropped_label = random_crop_3d(ct, mri, padded_label, patch_size)
        if cropped_label is None:
            raise RuntimeError("Internal crop error: label unexpectedly missing.")
        return cropped_ct, cropped_mri, cropped_label

    center = coords[np.random.randint(0, len(coords))]
    start = center - patch_size_arr // 2
    start = np.minimum(np.maximum(start, 0), np.maximum(shape - patch_size_arr, 0))
    slices = _crop_slices(start, patch_size_arr)
    return ct[slices], mri[slices], padded_label[slices]


def center_crop_3d(
    ct: np.ndarray,
    mri: np.ndarray,
    label: np.ndarray | None,
    patch_size: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Center crop a 3D CT/MR pair and optional label."""
    ct, mri, label = pad_to_patch_size(ct, mri, label, patch_size)
    patch_size_arr = np.asarray(patch_size, dtype=np.int64)
    shape = np.asarray(ct.shape, dtype=np.int64)
    start = np.maximum((shape - patch_size_arr) // 2, 0)
    slices = _crop_slices(start, patch_size_arr)
    return ct[slices], mri[slices], None if label is None else label[slices]
