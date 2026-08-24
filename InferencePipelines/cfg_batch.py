"""Utilities for keeping classifier-free guidance batches aligned."""

from __future__ import annotations

import torch


def expand_cfg_latents(latents: torch.Tensor, enabled: bool) -> torch.Tensor:
    """Repeat a latent batch in the ``[negative, positive]`` CFG order.

    SDXL prompt and pooled embeddings are concatenated as all unconditional
    rows followed by all conditional rows.  ``repeat_interleave(2)`` instead
    creates an interleaved order and silently pairs the wrong prompt whenever
    the effective batch contains more than one item.
    """

    if not enabled:
        return latents
    return torch.cat((latents, latents), dim=0)


def split_cfg_noise_pred(noise_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split UNet predictions in the matching ``[negative, positive]`` order."""

    return noise_pred.chunk(2)
