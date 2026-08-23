"""Weightless, model-free image metrics for the RepLDM eval pipeline.

These run in either conda env (pure numpy + PIL) and are deliberately
*decorrelated witnesses* for reward-hacking detection (EXPERIMENT_PLAN §13.5):
ImageReward / CLIP-family rewards co-move with saturation/colorfulness, so we
also report cheap statistics that expose whether an ImageReward gain is just a
global color/contrast push.

All functions take a uint8 RGB numpy array (H, W, 3) and return python floats.
`compute_all(pil_image)` is the entry point used by score.py.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _as_rgb_uint8(img: "Image.Image | np.ndarray") -> np.ndarray:
    if isinstance(img, Image.Image):
        img = np.asarray(img.convert("RGB"))
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img[..., :3].astype(np.uint8)


def colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colorfulness metric. Higher = more colorful.

    This is the axis ImageReward is biased toward and that scale*(TFSA-z) moves;
    a guidance gain that *only* raises this number is a reward-hacking signal.
    """
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


def laplacian_sharpness(rgb: np.ndarray) -> float:
    """Variance of the Laplacian on the luma channel (pure-numpy 3x3 kernel).

    A native-resolution high-frequency-energy proxy. Detail injected by guidance
    should move this; it survives the full-res image but is destroyed by the 224px
    downsample every learned reward applies (EXPERIMENT_PLAN §13.2).
    """
    luma = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    # 4-neighbour Laplacian via np.roll (no scipy dependency)
    lap = (
        -4.0 * luma
        + np.roll(luma, 1, axis=0)
        + np.roll(luma, -1, axis=0)
        + np.roll(luma, 1, axis=1)
        + np.roll(luma, -1, axis=1)
    )
    # drop the 1px border wrapped by np.roll
    lap = lap[1:-1, 1:-1]
    return float(lap.var())


def mean_saturation(rgb: np.ndarray) -> float:
    """Mean HSV saturation in [0, 1]."""
    x = rgb.astype(np.float32) / 255.0
    mx = x.max(axis=-1)
    mn = x.min(axis=-1)
    sat = np.where(mx > 1e-6, (mx - mn) / (mx + 1e-6), 0.0)
    return float(sat.mean())


def clipped_fraction(rgb: np.ndarray, lo: int = 2, hi: int = 253) -> float:
    """Fraction of pixels with any channel crushed to black or blown to white.

    Rises under over-saturation / over-contrast; a hacking co-indicator.
    """
    any_clipped = (rgb <= lo).any(axis=-1) | (rgb >= hi).any(axis=-1)
    return float(any_clipped.mean())


def contrast_std(rgb: np.ndarray) -> float:
    """Std of luma (global contrast)."""
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return float(luma.astype(np.float32).std())


def compute_all(img: "Image.Image | np.ndarray") -> dict:
    rgb = _as_rgb_uint8(img)
    return {
        "colorfulness": colorfulness(rgb),
        "laplacian_sharpness": laplacian_sharpness(rgb),
        "mean_saturation": mean_saturation(rgb),
        "clipped_fraction": clipped_fraction(rgb),
        "contrast_std": contrast_std(rgb),
    }


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        print(p, compute_all(Image.open(p)))
