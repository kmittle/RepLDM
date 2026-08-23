"""ImageReward scorer: global-IR (full image, 224 downsample) + patch-IR (native 224 crops).

Decoupled: imports ImageReward from the SOURCE checkout via the documented shim
(datasets stub + transformers symbol shim) — NOT the broken pip `image-reward`, and
NOT from Sana. patch-IR (native-res crops) is RepLDM's detail-sensitive twist (§13.2)
that Sana's ImageReward (full image only) lacks.
"""
import importlib.machinery
import os
import sys
import types

import numpy as np
from PIL import Image

from .base import Scorer, register_metric

IMAGEREWARD_SRC = "/mnt/miah204/bycao/ImageReward"
IR_CACHE = os.path.expanduser("~/.cache/ImageReward")


def _setup_imagereward_imports():
    if "datasets" not in sys.modules:
        d = types.ModuleType("datasets")
        d.load_dataset = lambda *a, **k: None
        d.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)
        sys.modules["datasets"] = d
    try:
        import transformers.modeling_utils as mu
        from transformers.pytorch_utils import (apply_chunking_to_forward,
                                                find_pruneable_heads_and_indices,
                                                prune_linear_layer)
        for n, f in [("apply_chunking_to_forward", apply_chunking_to_forward),
                     ("find_pruneable_heads_and_indices", find_pruneable_heads_and_indices),
                     ("prune_linear_layer", prune_linear_layer)]:
            if not hasattr(mu, n):
                setattr(mu, n, f)
    except Exception:
        pass
    if IMAGEREWARD_SRC not in sys.path:
        sys.path.insert(0, IMAGEREWARD_SRC)


def _native_crops(img, size=224):
    """Deterministic center + 4-corner crops at native resolution (no downsample)."""
    W, H = img.size
    if W < size or H < size:
        return [img.resize((size, size), Image.BICUBIC)]
    xs = [0, W - size, (W - size) // 2]
    ys = [0, H - size, (H - size) // 2]
    coords = [(xs[2], ys[2]), (xs[0], ys[0]), (xs[1], ys[0]), (xs[0], ys[1]), (xs[1], ys[1])]
    return [img.crop((x, y, x + size, y + size)) for (x, y) in coords]


@register_metric("imagereward")
class ImageRewardScorer(Scorer):
    OUTPUT_KEYS = (("imagereward", "higher"), ("patch_ir_mean", "higher"),
                   ("patch_ir_std", "witness"), ("patch_ir_n", "witness"))

    def __init__(self, device="cuda", patch_crops=5, no_patch_ir=False, **p):
        super().__init__(device, **p)
        self.patch_crops = patch_crops
        self.no_patch_ir = no_patch_ir
        _setup_imagereward_imports()
        import ImageReward
        self.model = ImageReward.load("ImageReward-v1.0", device=device, download_root=IR_CACHE)

    @classmethod
    def weights_status(cls, **p):
        ok = os.path.exists(os.path.join(IR_CACHE, "ImageReward.pt"))
        return ok, ("" if ok else f"missing {IR_CACHE}/ImageReward.pt")

    def score_image(self, image, prompt):
        rec = {"imagereward": float(self.model.score(prompt, image))}
        if not self.no_patch_ir:
            crops = _native_crops(image, 224)
            if self.patch_crops:
                crops = crops[:self.patch_crops]
            ps = [float(self.model.score(prompt, c)) for c in crops]
            rec["patch_ir_mean"] = float(np.mean(ps))
            rec["patch_ir_std"] = float(np.std(ps))
            rec["patch_ir_n"] = len(crops)
        return rec
