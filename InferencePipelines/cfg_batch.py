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


def expand_cfg_time_ids(
    time_ids: torch.Tensor, batch_size: int, enabled: bool
) -> torch.Tensor:
    """Repeat one SDXL time-id row per effective sample and CFG branch.

    ``time_ids`` must contain one row when CFG is disabled and one negative plus
    one positive row when it is enabled. Repeating the branch block preserves
    the same all-negative/all-positive order as prompt and pooled embeddings.
    """

    if not isinstance(time_ids, torch.Tensor) or time_ids.ndim != 2:
        raise ValueError("time_ids must be a rank-2 tensor")
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer")
    batch_size = int(batch_size)
    branch_rows = 2 if enabled else 1
    if time_ids.shape[0] != branch_rows:
        raise ValueError(
            f"time_ids must contain {branch_rows} branch rows, got {time_ids.shape[0]}"
        )
    return time_ids.repeat_interleave(batch_size, dim=0)
