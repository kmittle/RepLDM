"""Same-NFE sparse/dense cross-attention baselines used by the S5 audit."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import re
from typing import Dict, Iterator, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


PLADIS_SOURCE_COMMIT = "248b9d15701c08094c47dc90b4ae24afbf5cf7a9"
PLADIS_OPERATOR_PORT_IMPLEMENTATION = "pladis_operator_port_248b9d1"
PLADIS_UPSTREAM_DIFFUSERS_VERSION = "0.33.1"
PLADIS_PORT_DIFFUSERS_VERSION = "0.32.1"
PLADIS_PINNED_SDXL_LAYERS = ("up", "down")
PLADIS_PINNED_PROBABILITY_DTYPE = "query"
PLADIS_PINNED_SDXL_GROUP_COUNTS = {"up": 36, "down": 24}
PLADIS_PINNED_SDXL_PROCESSOR_COUNT = 60
PLADIS_PINNED_SDXL_PROCESSOR_NAMES_SHA256 = (
    "2d66ed06dfc07e6d0b0d7ce1a8d39bd262c5e5e86b5d1947e1e412f5b6fe8c8f"
)
GAG_EQ13_IMPLEMENTATION = "gag_eq13_reimplementation_2603.02531v2"
GAG_PAPER_ID = "2603.02531v2"
GAG_PAPER_EQUATIONS = (12, 13)
GAG_INHERITED_LAYER_POLICY = PLADIS_PINNED_SDXL_LAYERS
GAG_INHERITED_PROBABILITY_DTYPE = PLADIS_PINNED_PROBABILITY_DTYPE
ATTENTION_BASELINE_LAYER_GROUPS = frozenset({"down", "mid", "up"})
ATTENTION_PROBABILITY_DTYPES = frozenset({"float32", "query"})
ATTENTION_MASK_POLICIES = frozenset({"supported", "none"})


def attention_processor_names_sha256(names: Sequence[str]) -> str:
    """Hash the sorted processor names using the registered compact-JSON rule."""
    payload = json.dumps(
        sorted(str(name) for name in names),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _entmax15(logits: Tensor) -> Tensor:
    try:
        from entmax import entmax15
    except ImportError as exc:  # pragma: no cover - exercised in an unprepared env
        raise ImportError(
            "PLADIS/GAG baselines require the optional 'entmax' package"
        ) from exc
    return entmax15(logits, dim=-1)


def _gag_parallel_component(
    sparse_output: Tensor, dense_output: Tensor
) -> Tensor:
    """Project the sparse-dense residual onto the sparse output (Eq. 13)."""
    sparse_geometry = sparse_output.float()
    residual_attention = sparse_geometry - dense_output.float()
    parallel_denom = sparse_geometry.square().sum(dim=-1, keepdim=True)
    safe_parallel_denom = torch.where(
        parallel_denom != 0,
        parallel_denom,
        torch.ones_like(parallel_denom),
    )
    return (
        (residual_attention * sparse_geometry).sum(dim=-1, keepdim=True)
        / safe_parallel_denom
    ) * sparse_geometry


def _pladis_guided_output(
    sparse: Tensor, dense: Tensor, value: Tensor, attention_scale: float
) -> Tensor:
    """Apply the pinned sparse/dense PLADIS mixture before output projection."""
    mixed = float(attention_scale) * sparse + (1.0 - float(attention_scale)) * dense
    return torch.matmul(mixed, value)


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
        probability_dtype: str = "float32",
        attention_mask_policy: str = "supported",
    ) -> None:
        if kind not in {"pladis", "gag"}:
            raise ValueError(f"unsupported sparse/dense baseline {kind!r}")
        if alpha != 1.5:
            raise ValueError("the registered S5 baseline uses alpha-entmax alpha=1.5")
        if attention_scale < 0:
            raise ValueError("attention baseline scale must be non-negative")
        if kind == "gag" and (eta <= 0 or not 0 <= zeta <= 1):
            raise ValueError("GAG requires eta > 0 and zeta in [0, 1]")
        if probability_dtype not in ATTENTION_PROBABILITY_DTYPES:
            raise ValueError(
                "probability_dtype must be 'float32' or 'query'"
            )
        if attention_mask_policy not in ATTENTION_MASK_POLICIES:
            raise ValueError("attention_mask_policy must be 'supported' or 'none'")
        self.kind = kind
        self.attention_scale = float(attention_scale)
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.zeta = float(zeta)
        self.probability_dtype = str(probability_dtype)
        self.attention_mask_policy = str(attention_mask_policy)
        self.call_count = 0

    @staticmethod
    def _prepare_mask(attn, attention_mask, key_length, batch_size):
        if attention_mask is None:
            return None
        mask = attn.prepare_attention_mask(attention_mask, key_length, batch_size)
        return mask.view(batch_size, attn.heads, -1, mask.shape[-1])

    def _weights(
        self, query: Tensor, key: Tensor, attention_mask: Optional[Tensor]
    ) -> tuple[Tensor, Tensor]:
        head_dim = query.shape[-1]
        logits = torch.matmul(query, key.transpose(-1, -2)) * (head_dim ** -0.5)
        if attention_mask is not None and attention_mask.dtype == torch.bool:
            logits = logits.masked_fill(~attention_mask, -torch.inf)
        elif attention_mask is not None:
            logits = logits + attention_mask
        probability_logits = (
            logits.float() if self.probability_dtype == "float32" else logits
        )
        dense = torch.softmax(probability_logits, dim=-1).to(query.dtype)
        sparse = _entmax15(probability_logits).to(query.dtype)
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
        self.call_count += 1
        if attention_mask is not None and self.attention_mask_policy == "none":
            raise ValueError("registered attention baseline requires attention_mask=None")
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        original_shape = None
        if input_ndim == 4:
            batch, channels, height, width = hidden_states.shape
            original_shape = (batch, channels, height, width)
            hidden_states = hidden_states.view(batch, channels, height * width).transpose(1, 2)
        batch_size = hidden_states.shape[0]
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        mask = self._prepare_mask(attn, attention_mask, key.shape[1], batch_size)
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
            output = _pladis_guided_output(
                sparse, dense, value, self.attention_scale
            )
        else:
            sparse_output = torch.matmul(sparse, value)
            dense_output = torch.matmul(dense, value)
            sparse_geometry = sparse_output.float()
            residual_attention = sparse_geometry - dense_output.float()
            parallel = _gag_parallel_component(sparse_output, dense_output)
            geometry = parallel + self.zeta * (residual_attention - parallel)
            geometry_norm = torch.linalg.vector_norm(geometry, dim=-1, keepdim=True)
            cap = torch.clamp(self.eta / geometry_norm.clamp_min(torch.finfo(torch.float32).eps), max=1.0)
            output = sparse_geometry + self.attention_scale * geometry * cap
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
    applied_layers: Optional[Sequence[str]] = None,
    probability_dtype: str = "float32",
    expected_group_counts: Optional[Mapping[str, int]] = None,
    expected_processor_names_sha256: Optional[str] = None,
    attention_mask_policy: str = "supported",
) -> Iterator[dict]:
    """Install a baseline on all cross-attention processors and always restore it."""
    normalized_layers = None
    if applied_layers is not None:
        if isinstance(applied_layers, (str, bytes)):
            raise ValueError("applied_layers must be a sequence of layer groups")
        normalized_layers = tuple(str(value) for value in applied_layers)
        if not normalized_layers or len(normalized_layers) != len(set(normalized_layers)):
            raise ValueError("applied_layers must be non-empty and unique")
        unknown = set(normalized_layers) - ATTENTION_BASELINE_LAYER_GROUPS
        if unknown:
            raise ValueError(f"unsupported attention layer groups: {sorted(unknown)}")

    def layer_group(name: str) -> Optional[str]:
        if name.startswith("down_blocks."):
            return "down"
        if name.startswith("mid_block."):
            return "mid"
        if name.startswith("up_blocks."):
            return "up"
        return None

    originals = dict(unet.attn_processors)
    replacements: Dict[str, object] = dict(originals)
    matched_names = []
    observed_group_counts = {group: 0 for group in ATTENTION_BASELINE_LAYER_GROUPS}
    for name in originals:
        if name.endswith("attn2.processor") and (
            normalized_layers is None or layer_group(name) in normalized_layers
        ):
            replacements[name] = SparseDenseAttentionProcessor(
                kind=kind,
                attention_scale=attention_scale,
                alpha=alpha,
                eta=eta,
                zeta=zeta,
                probability_dtype=probability_dtype,
                attention_mask_policy=attention_mask_policy,
            )
            matched_names.append(name)
            group = layer_group(name)
            if group is not None:
                observed_group_counts[group] += 1
    if not matched_names:
        raise ValueError("no cross-attention processors matched '*attn2.processor'")
    if normalized_layers is not None:
        missing_groups = [
            group for group in normalized_layers if observed_group_counts[group] == 0
        ]
        if missing_groups:
            raise ValueError(
                "no cross-attention processors matched requested layer groups: "
                f"{missing_groups}"
            )
    if expected_group_counts is not None:
        if not isinstance(expected_group_counts, Mapping):
            raise ValueError("expected_group_counts must be a mapping")
        normalized_expected_counts = {}
        for raw_group, raw_count in expected_group_counts.items():
            group = str(raw_group)
            if group not in ATTENTION_BASELINE_LAYER_GROUPS:
                raise ValueError(f"unsupported expected layer group {group!r}")
            if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count <= 0:
                raise ValueError("expected processor group counts must be positive integers")
            normalized_expected_counts[group] = raw_count
        expected_groups = set(normalized_layers or ATTENTION_BASELINE_LAYER_GROUPS)
        if set(normalized_expected_counts) != expected_groups:
            raise ValueError(
                "expected processor group counts must exactly cover applied_layers"
            )
        observed_selected_counts = {
            group: observed_group_counts[group] for group in normalized_expected_counts
        }
        if observed_selected_counts != normalized_expected_counts:
            raise ValueError(
                "cross-attention processor group counts differ from the registered "
                f"topology: expected {normalized_expected_counts}, observed "
                f"{observed_selected_counts}"
            )
    observed_names_sha256 = attention_processor_names_sha256(matched_names)
    if expected_processor_names_sha256 is not None:
        if not isinstance(expected_processor_names_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_processor_names_sha256
        ):
            raise ValueError(
                "expected_processor_names_sha256 must be 64 lowercase hex"
            )
        if observed_names_sha256 != expected_processor_names_sha256:
            raise ValueError(
                "cross-attention processor names differ from the registered topology"
            )
    selected_processors = {
        name: replacements[name] for name in matched_names
    }
    topology = {
        "group_counts": {
            group: observed_group_counts[group]
            for group in (normalized_layers or sorted(ATTENTION_BASELINE_LAYER_GROUPS))
        },
        "processor_count": len(matched_names),
        "processor_names_sha256": observed_names_sha256,
        "processors_called": 0,
        "processor_calls_total": 0,
        "processor_call_count_min": 0,
        "processor_call_count_max": 0,
    }
    unet.set_attn_processor(replacements)
    try:
        yield topology
    finally:
        call_counts = [processor.call_count for processor in selected_processors.values()]
        topology.update(
            {
                "processors_called": sum(count > 0 for count in call_counts),
                "processor_calls_total": sum(call_counts),
                "processor_call_count_min": min(call_counts),
                "processor_call_count_max": max(call_counts),
            }
        )
        unet.set_attn_processor(dict(originals))
