from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


ArrayTriplet = Tuple[np.ndarray, np.ndarray, np.ndarray | None]


@dataclass(frozen=True)
class AugmentationConfig:
    flip_prob: float = 0.5
    intensity_scale_prob: float = 0.15
    intensity_scale_range: tuple[float, float] = (0.9, 1.1)
    intensity_shift_prob: float = 0.15
    intensity_shift_range: tuple[float, float] = (-0.1, 0.1)
    gaussian_noise_prob: float = 0.15
    gaussian_noise_std: float = 0.01
    gamma_prob: float = 0.15
    gamma_range: tuple[float, float] = (0.7, 1.5)


def random_flip_3d(
    ct: np.ndarray,
    mri: np.ndarray,
    label: np.ndarray | None = None,
    axes: Sequence[int] = (0, 1, 2),
    prob: float = 0.5,
    rng: np.random.Generator | None = None,
) -> ArrayTriplet:
    """Synchronously flip CT, MR, and optional label along spatial axes."""
    rng = np.random.default_rng() if rng is None else rng
    if ct.shape != mri.shape:
        raise ValueError(f"CT/MR shape mismatch: {ct.shape} vs {mri.shape}.")
    if label is not None and label.shape != ct.shape:
        raise ValueError(f"Image/label shape mismatch: image={ct.shape}, label={label.shape}.")

    for axis in axes:
        if rng.random() < prob:
            ct = np.flip(ct, axis=axis)
            mri = np.flip(mri, axis=axis)
            if label is not None:
                label = np.flip(label, axis=axis)
    return ct.copy(), mri.copy(), None if label is None else label.copy()


def random_intensity_scale_shift(
    image: np.ndarray,
    scale_range: tuple[float, float] = (0.9, 1.1),
    shift_range: tuple[float, float] = (-0.1, 0.1),
    scale_prob: float = 0.15,
    shift_prob: float = 0.15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply random multiplicative/additive intensity perturbation."""
    rng = np.random.default_rng() if rng is None else rng
    out = image.astype(np.float32, copy=True)
    if rng.random() < scale_prob:
        out *= float(rng.uniform(scale_range[0], scale_range[1]))
    if rng.random() < shift_prob:
        out += float(rng.uniform(shift_range[0], shift_range[1]))
    return out.astype(np.float32)


def random_gaussian_noise(
    image: np.ndarray,
    std: float = 0.01,
    prob: float = 0.15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add zero-mean Gaussian noise with probability ``prob``."""
    rng = np.random.default_rng() if rng is None else rng
    out = image.astype(np.float32, copy=True)
    if rng.random() < prob:
        out += rng.normal(0.0, std, size=out.shape).astype(np.float32)
    return out.astype(np.float32)


def random_gamma(
    image: np.ndarray,
    gamma_range: tuple[float, float] = (0.7, 1.5),
    prob: float = 0.15,
    retain_stats: bool = True,
    eps: float = 1e-8,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply gamma augmentation after min-max scaling, then restore statistics."""
    rng = np.random.default_rng() if rng is None else rng
    out = image.astype(np.float32, copy=True)
    if rng.random() >= prob:
        return out

    mean = float(out.mean())
    std = float(out.std())
    min_val = float(out.min())
    max_val = float(out.max())
    if max_val - min_val < eps:
        return out

    gamma = float(rng.uniform(gamma_range[0], gamma_range[1]))
    scaled = (out - min_val) / (max_val - min_val + eps)
    scaled = np.power(np.clip(scaled, 0.0, 1.0), gamma)
    out = scaled * (max_val - min_val) + min_val
    if retain_stats:
        out = (out - out.mean()) / (out.std() + eps)
        out = out * std + mean
    return out.astype(np.float32)


def apply_intensity_augmentations(
    ct: np.ndarray,
    mri: np.ndarray,
    config: AugmentationConfig = AugmentationConfig(),
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply image-only intensity augmentations to CT and MR."""
    rng = np.random.default_rng() if rng is None else rng
    ct = random_intensity_scale_shift(
        ct,
        scale_range=config.intensity_scale_range,
        shift_range=config.intensity_shift_range,
        scale_prob=config.intensity_scale_prob,
        shift_prob=config.intensity_shift_prob,
        rng=rng,
    )
    mri = random_intensity_scale_shift(
        mri,
        scale_range=config.intensity_scale_range,
        shift_range=config.intensity_shift_range,
        scale_prob=config.intensity_scale_prob,
        shift_prob=config.intensity_shift_prob,
        rng=rng,
    )
    ct = random_gaussian_noise(ct, std=config.gaussian_noise_std, prob=config.gaussian_noise_prob, rng=rng)
    mri = random_gaussian_noise(mri, std=config.gaussian_noise_std, prob=config.gaussian_noise_prob, rng=rng)
    ct = random_gamma(ct, gamma_range=config.gamma_range, prob=config.gamma_prob, rng=rng)
    mri = random_gamma(mri, gamma_range=config.gamma_range, prob=config.gamma_prob, rng=rng)
    return ct.astype(np.float32), mri.astype(np.float32)


def augment_ct_mr_3d(
    ct: np.ndarray,
    mri: np.ndarray,
    label: np.ndarray | None = None,
    config: AugmentationConfig = AugmentationConfig(),
    rng: np.random.Generator | None = None,
) -> ArrayTriplet:
    """Apply lightweight CT/MR 3D augmentations.

    Spatial flips are synchronized across CT, MR, and label. Intensity
    perturbations are applied only to CT/MR images.
    """
    rng = np.random.default_rng() if rng is None else rng
    ct, mri, label = random_flip_3d(ct, mri, label, prob=config.flip_prob, rng=rng)
    ct, mri = apply_intensity_augmentations(ct, mri, config=config, rng=rng)
    return ct, mri, label
