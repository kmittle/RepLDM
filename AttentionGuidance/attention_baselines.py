"""Same-NFE sparse/dense cross-attention baselines used by the S5 audit."""
from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Dict, Iterator, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def _entmax15(logits: Tensor) -> Tensor:
    try:
        from entmax import entmax15
    except ImportError as exc:  # pragma: no cover - exercised in an unprepared env
        raise ImportError(
            "PLADIS/GAG baselines require the optional 'entmax' package"
        ) from exc
    return entmax15(logits, dim=-1)


class SparseDenseAttentionProcessor:
    """Implement PLADIS or GAG inside an otherwise ordinary attention processor."""

    def __init__(
        self,
        *,
        kind: str,
        attention_scale: float,
        alpha: float = 1.5,
        eta: float = 15.0,
        zeta: float = 0.0,
    ) -> None:
        if kind not in {"pladis", "gag"}:
            raise ValueError(f"unsupported sparse/dense baseline {kind!r}")
        if alpha != 1.5:
            raise ValueError("the registered S5 baseline uses alpha-entmax alpha=1.5")
        if attention_scale < 0:
            raise ValueError("attention baseline scale must be non-negative")
        if kind == "gag" and (eta <= 0 or not 0 <= zeta <= 1):
            raise ValueError("GAG requires eta > 0 and zeta in [0, 1]")
        self.kind = kind
        self.attention_scale = float(attention_scale)
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.zeta = float(zeta)

    @staticmethod
    def _prepare_mask(attn, attention_mask, sequence_length, batch_size):
        if attention_mask is None:
            return None
        mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        return mask.view(batch_size, attn.heads, -1, mask.shape[-1])

    @staticmethod
    def _weights(query: Tensor, key: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        head_dim = query.shape[-1]
        logits = torch.matmul(query, key.transpose(-1, -2)) * (head_dim ** -0.5)
        if attention_mask is not None:
            logits = logits + attention_mask
        dense = torch.softmax(logits.float(), dim=-1).to(query.dtype)
        sparse = _entmax15(logits.float()).to(query.dtype)
        return sparse, dense

    @staticmethod
    def _finish(
        attn,
        hidden_states: Tensor,
        residual: Tensor,
        input_ndim: int,
        batch_size: int,
        head_dim: int,
        original_shape,
    ):
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(residual.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                original_shape[0], original_shape[1], original_shape[2], original_shape[3]
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor

    def __call__(
        self,
        attn,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        temb: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        del args, kwargs
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        original_shape = None
        if input_ndim == 4:
            batch, channels, height, width = hidden_states.shape
            original_shape = (batch, channels, height, width)
            hidden_states = hidden_states.view(batch, channels, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape
        mask = self._prepare_mask(attn, attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        norm_q = getattr(attn, "norm_q", None)
        norm_k = getattr(attn, "norm_k", None)
        if norm_q is not None:
            query = norm_q(query)
        if norm_k is not None:
            key = norm_k(key)
        sparse, dense = self._weights(query, key, mask)
        if self.kind == "pladis":
            mixed = self.attention_scale * sparse + (1.0 - self.attention_scale) * dense
            output = torch.matmul(mixed, value)
        else:
            sparse_output = torch.matmul(sparse, value)
            dense_output = torch.matmul(dense, value)
            residual_attention = sparse_output - dense_output
            parallel_denom = sparse_output.square().sum(dim=-1, keepdim=True)
            parallel = (
                (residual_attention * sparse_output).sum(dim=-1, keepdim=True)
                / parallel_denom.clamp_min(torch.finfo(sparse_output.dtype).eps)
            ) * sparse_output
            geometry = parallel + self.zeta * (residual_attention - parallel)
            geometry_norm = torch.linalg.vector_norm(geometry.float(), dim=-1, keepdim=True)
            cap = torch.clamp(self.eta / geometry_norm.clamp_min(torch.finfo(torch.float32).eps), max=1.0)
            output = sparse_output + self.attention_scale * geometry * cap.to(geometry.dtype)
        return self._finish(
            attn,
            output,
            residual,
            input_ndim,
            batch_size,
            head_dim,
            original_shape,
        )


@contextmanager
def installed_attention_baseline(
    unet: torch.nn.Module,
    *,
    kind: str,
    attention_scale: float,
    alpha: float = 1.5,
    eta: float = 15.0,
    zeta: float = 0.0,
) -> Iterator[None]:
    """Install a baseline on all cross-attention processors and always restore it."""
    originals = dict(unet.attn_processors)
    replacements: Dict[str, object] = dict(originals)
    replacement_count = 0
    for name in originals:
        if name.endswith("attn2.processor"):
            replacements[name] = SparseDenseAttentionProcessor(
                kind=kind,
                attention_scale=attention_scale,
                alpha=alpha,
                eta=eta,
                zeta=zeta,
            )
            replacement_count += 1
    if replacement_count == 0:
        raise ValueError("no cross-attention processors matched '*attn2.processor'")
    unet.set_attn_processor(replacements)
    try:
        yield
    finally:
        unet.set_attn_processor(dict(originals))
