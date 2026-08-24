"""Scheduler-consistent transport from frozen self-attention and ``x_0``.

The module deliberately has no diffusers dependency.  The pipeline supplies
the Q/K projections captured from one ordinary UNet forward and the
``pred_original_sample`` returned by its scheduler.  No processor is replaced
for the semantic transport path.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


SEMANTIC_TRANSPORT_MODES = frozenset(
    {
        "clean_tfsa",
        "reciprocal_latent",
        "reciprocal_semantic",
        "reciprocal_semantic_permuted",
    }
)


@dataclass(frozen=True)
class SemanticTransportConfig:
    """Frozen inference-time parameters for one semantic transport action."""

    mode: str
    angle: float
    topk: int = 16
    layer_name: str = "up_blocks.0.attentions.0.transformer_blocks.0.attn1"
    permutation_seed: int = 1729

    def __post_init__(self) -> None:
        if self.mode not in SEMANTIC_TRANSPORT_MODES:
            raise ValueError(f"unsupported semantic transport mode {self.mode!r}")
        if not math.isfinite(float(self.angle)) or self.angle < 0:
            raise ValueError("semantic transport angle must be finite and non-negative")
        if self.mode == "clean_tfsa":
            return
        if int(self.topk) <= 0:
            raise ValueError("semantic transport topk must be positive")


def resolve_module(root: torch.nn.Module, dotted_name: str) -> torch.nn.Module:
    """Resolve a dotted module path and fail loudly on a layer typo."""
    module: torch.nn.Module = root
    for part in dotted_name.split("."):
        if not hasattr(module, part):
            raise AttributeError(f"module path {dotted_name!r} missing component {part!r}")
        module = getattr(module, part)
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"module path {dotted_name!r} component {part!r} is not a module")
    return module


class QKCapture:
    """Temporarily observe ``to_q`` and ``to_k`` without changing outputs."""

    def __init__(self, attention_module: torch.nn.Module) -> None:
        if not hasattr(attention_module, "to_q") or not hasattr(attention_module, "to_k"):
            raise TypeError("semantic transport layer must expose to_q and to_k")
        self.attention_module = attention_module
        self.query: Optional[Tensor] = None
        self.key: Optional[Tensor] = None

    @staticmethod
    def _tensor_output(output) -> Tensor:
        value = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(value, Tensor):
            raise TypeError("Q/K hook output must be a tensor")
        return value.detach()

    @contextmanager
    def forward(self) -> Iterator["QKCapture"]:
        self.query = None
        self.key = None
        query_handle = self.attention_module.to_q.register_forward_hook(
            lambda _module, _inputs, output: setattr(self, "query", self._tensor_output(output))
        )
        key_handle = self.attention_module.to_k.register_forward_hook(
            lambda _module, _inputs, output: setattr(self, "key", self._tensor_output(output))
        )
        try:
            yield self
        finally:
            query_handle.remove()
            key_handle.remove()

    def get_conditional(
        self,
        *,
        do_classifier_free_guidance: bool,
        batch_size: int,
    ) -> Tuple[Tensor, Tensor]:
        if self.query is None or self.key is None:
            raise RuntimeError("Q/K projections were not captured by the UNet forward")
        if self.query.shape[0] != self.key.shape[0]:
            raise ValueError("captured Q and K batch dimensions differ")
        if not do_classifier_free_guidance:
            return self.query, self.key
        expected = 2 * batch_size
        if self.query.shape[0] != expected:
            raise ValueError(
                f"expected {expected} Q/K rows for CFG, got {self.query.shape[0]}"
            )
        # The SDXL pipeline concatenates negative rows before positive rows.
        return self.query[batch_size:], self.key[batch_size:]


def infer_token_grid(num_tokens: int, height: int, width: int) -> Tuple[int, int]:
    """Choose a token grid with the latent aspect ratio and exact token count."""
    if num_tokens <= 0 or height <= 0 or width <= 0:
        raise ValueError("token count and spatial dimensions must be positive")
    target_height = math.sqrt(num_tokens * height / width)
    divisors = [d for d in range(1, int(math.sqrt(num_tokens)) + 1) if num_tokens % d == 0]
    if not divisors:
        raise ValueError(f"cannot factor token count {num_tokens}")
    height_candidates = divisors + [num_tokens // d for d in divisors]
    grid_height = min(height_candidates, key=lambda d: abs(d - target_height))
    return int(grid_height), int(num_tokens // grid_height)


def _reshape_heads(value: Tensor, heads: int) -> Tensor:
    if value.ndim != 3:
        raise ValueError("captured Q/K tensors must have shape (batch, tokens, channels)")
    batch, tokens, channels = value.shape
    if channels % heads:
        raise ValueError(f"projection channels {channels} are not divisible by {heads} heads")
    return value.reshape(batch, tokens, heads, channels // heads).transpose(1, 2)


def mean_head_attention(
    query: Tensor,
    key: Tensor,
    *,
    heads: int,
    norm_q: Optional[torch.nn.Module] = None,
    norm_k: Optional[torch.nn.Module] = None,
    head_chunk: int = 4,
) -> Tensor:
    """Compute the mean dense attention map in float32 in bounded head chunks."""
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError("Q and K must have matching batch, token, and channel dimensions")
    q_heads = _reshape_heads(query, heads)
    k_heads = _reshape_heads(key, heads)
    if norm_q is not None:
        q_heads = norm_q(q_heads)
    if norm_k is not None:
        k_heads = norm_k(k_heads)
    batch, num_heads, tokens, head_dim = q_heads.shape
    result = torch.zeros((batch, tokens, tokens), device=query.device, dtype=torch.float32)
    scale = head_dim ** -0.5
    for start in range(0, num_heads, max(1, int(head_chunk))):
        stop = min(num_heads, start + max(1, int(head_chunk)))
        logits = torch.matmul(
            q_heads[:, start:stop].float(), k_heads[:, start:stop].float().transpose(-1, -2)
        ) * scale
        result += torch.softmax(logits, dim=-1).sum(dim=1)
    result /= float(num_heads)
    if not torch.isfinite(result).all():
        raise FloatingPointError("non-finite self-attention affinity")
    return result


def _mutual_reciprocal_affinity(
    affinity: Tensor,
    *,
    topk: int,
    permuted: bool = False,
    permutation_seed: int = 1729,
) -> Tensor:
    """Build a row-stochastic reciprocal graph with mutual top-k support."""
    if affinity.ndim != 3 or affinity.shape[-1] != affinity.shape[-2]:
        raise ValueError("affinity must have shape (batch, tokens, tokens)")
    tokens = affinity.shape[-1]
    reciprocal = torch.sqrt(
        torch.clamp(affinity * affinity.transpose(-1, -2), min=0.0)
    )
    k = min(max(1, int(topk)), tokens)
    indices = reciprocal.topk(k, dim=-1, largest=True, sorted=False).indices
    support = torch.zeros_like(reciprocal, dtype=torch.bool)
    support.scatter_(-1, indices, True)
    support = support & support.transpose(-1, -2)
    diagonal = torch.eye(tokens, device=affinity.device, dtype=torch.bool).unsqueeze(0)
    support = support | diagonal
    graph = reciprocal.masked_fill(~support, 0.0)
    graph = graph / graph.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(graph.dtype).eps)
    if permuted:
        permutation = deterministic_permutation(tokens, permutation_seed, affinity.device)
        graph = graph.index_select(-2, permutation).index_select(-1, permutation)
    if not torch.isfinite(graph).all():
        raise FloatingPointError("non-finite reciprocal affinity")
    return graph


def deterministic_permutation(tokens: int, seed: int, device: torch.device) -> Tensor:
    """Return a deterministic bijection without consuming a sampling generator."""
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    # An affine permutation is stable across devices and preserves the graph's
    # value multiset while making token-to-spatial-position alignment wrong.
    factor = tokens // 2 + 1
    while math.gcd(factor, tokens) != 1:
        factor += 1
    offset = (int(seed) * 2654435761) % tokens
    return (torch.arange(tokens, device=device, dtype=torch.long) * factor + offset) % tokens


def affinity_from_tokens(
    tokens: Tensor,
    *,
    mode: str,
    topk: int,
    permutation_seed: int = 1729,
) -> Tuple[Tensor, Tensor]:
    """Return transport weights and normalized dense-affinity entropy."""
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape (batch, tokens, channels)")
    work = tokens.float()
    logits = torch.matmul(work / math.sqrt(work.shape[-1]), work.transpose(-1, -2))
    affinity = torch.softmax(logits, dim=-1)
    entropy = -(
        affinity.clamp_min(torch.finfo(affinity.dtype).tiny) * affinity.clamp_min(
            torch.finfo(affinity.dtype).tiny
        ).log()
    ).sum(dim=-1).mean(dim=-1) / math.log(affinity.shape[-1])
    if mode == "clean_tfsa":
        graph = affinity
    else:
        graph = _mutual_reciprocal_affinity(
            affinity,
            topk=topk,
            permuted=mode == "reciprocal_semantic_permuted",
            permutation_seed=permutation_seed,
        )
    return graph, entropy


def affinity_from_qk(
    query: Tensor,
    key: Tensor,
    *,
    attention_module: torch.nn.Module,
    mode: str,
    topk: int,
    permutation_seed: int = 1729,
) -> Tuple[Tensor, Tensor]:
    heads = int(getattr(attention_module, "heads"))
    affinity = mean_head_attention(
        query,
        key,
        heads=heads,
        norm_q=getattr(attention_module, "norm_q", None),
        norm_k=getattr(attention_module, "norm_k", None),
    )
    entropy = -(
        affinity.clamp_min(torch.finfo(affinity.dtype).tiny)
        * affinity.clamp_min(torch.finfo(affinity.dtype).tiny).log()
    ).sum(dim=-1).mean(dim=-1) / math.log(affinity.shape[-1])
    graph = _mutual_reciprocal_affinity(
        affinity,
        topk=topk,
        permuted=mode == "reciprocal_semantic_permuted",
        permutation_seed=permutation_seed,
    )
    return graph, entropy


def fixed_moment_transport(
    x0: Tensor,
    graph: Tensor,
    *,
    angle: float,
    confidence: Tensor,
    grid_height: int,
    grid_width: int,
) -> Tensor:
    """Move ``x0`` on each channel's fixed-mean/fixed-variance sphere."""
    if x0.ndim != 4:
        raise ValueError("x0 must have shape (batch, channels, height, width)")
    batch, channels, height, width = x0.shape
    if graph.shape != (batch, grid_height * grid_width, grid_height * grid_width):
        raise ValueError("graph shape does not match the requested transport grid")
    if confidence.ndim != 1 or confidence.shape[0] != batch:
        raise ValueError("confidence must have one value per x0 batch item")
    if float(angle) == 0.0:
        return x0
    work_dtype = (
        torch.float32 if x0.dtype in {torch.float16, torch.bfloat16} else x0.dtype
    )
    work = x0.to(work_dtype)
    coarse = F.interpolate(work, size=(grid_height, grid_width), mode="area")
    tokens = coarse.flatten(2).transpose(1, 2)
    moved_tokens = torch.bmm(graph.to(work.dtype), tokens)
    coarse_residual = (moved_tokens - tokens).transpose(1, 2).reshape(
        batch, channels, grid_height, grid_width
    )
    residual = F.interpolate(
        coarse_residual, size=(height, width), mode="bilinear", align_corners=False
    )

    spatial = (-2, -1)
    mean = work.mean(dim=spatial, keepdim=True)
    centered = work - mean
    centered_residual = residual - residual.mean(dim=spatial, keepdim=True)
    energy_epsilon = torch.finfo(work.dtype).eps
    latent_energy = centered.square().sum(dim=spatial, keepdim=True)
    tangent = centered_residual - (
        (centered_residual * centered).sum(dim=spatial, keepdim=True)
        / latent_energy.clamp_min(energy_epsilon)
    ) * centered
    tangent_energy = tangent.square().sum(dim=spatial, keepdim=True)
    radius = torch.sqrt(latent_energy.clamp_min(0.0))
    speed = torch.sqrt(tangent_energy.clamp_min(0.0))
    direction = tangent / speed.clamp_min(math.sqrt(energy_epsilon))
    angle_tensor = float(angle) * confidence.to(work).reshape(batch, 1, 1, 1)
    moved = (
        torch.cos(angle_tensor * speed / radius.clamp_min(math.sqrt(energy_epsilon))) * centered
        + torch.sin(angle_tensor * speed / radius.clamp_min(math.sqrt(energy_epsilon)))
        * radius
        * direction
        + mean
    )
    active = (latent_energy > energy_epsilon) & (tangent_energy > energy_epsilon)
    return torch.where(active, moved, work)


def inject_predicted_clean_update(
    prev_sample: Tensor, pred_original_sample: Tensor, guided_x0: Tensor
) -> Tensor:
    """Apply ``guided_x0 - pred_original_sample`` after one scheduler step.

    Keeping this tiny operation explicit prevents callers from silently
    rebuilding Euler's ``x_0`` estimate with a different dtype or scheduler.
    """
    update = guided_x0 - pred_original_sample
    return (prev_sample + update).to(prev_sample.dtype)


class SemanticTransport:
    """Stateful per-rollout transport and diagnostics collector."""

    def __init__(
        self,
        unet: torch.nn.Module,
        config: SemanticTransportConfig,
        *,
        batch_size: int,
        do_classifier_free_guidance: bool,
    ) -> None:
        self.config = config
        self.batch_size = int(batch_size)
        self.do_classifier_free_guidance = bool(do_classifier_free_guidance)
        self.attention_module = (
            resolve_module(unet, config.layer_name)
            if "semantic" in config.mode
            else None
        )
        self.capture = QKCapture(self.attention_module) if self.attention_module else None
        self._entropy = []
        self._confidence = []
        self._ratios = []
        self._transport_norms = []
        self._scheduler_norms = []

    @contextmanager
    def capture_forward(self) -> Iterator[None]:
        if self.capture is None:
            yield
        else:
            with self.capture.forward():
                yield

    def apply(
        self,
        *,
        step_output,
        scheduler_update: Tensor,
    ) -> Tensor:
        if self.config.angle == 0:
            return step_output.prev_sample
        x0 = step_output.pred_original_sample
        prev_sample = step_output.prev_sample
        batch, _channels, height, width = x0.shape
        if self.config.mode == "clean_tfsa" or self.config.mode == "reciprocal_latent":
            # The latent-grid target is fixed at 32x32 for the 1024² SDXL run.
            coarse_height = min(height, max(1, round(height / 4)))
            coarse_width = min(width, max(1, round(width / 4)))
            tokens = F.interpolate(
                x0.float(), size=(coarse_height, coarse_width), mode="area"
            ).flatten(2).transpose(1, 2)
            graph, entropy = affinity_from_tokens(
                tokens,
                mode=self.config.mode,
                topk=self.config.topk,
                permutation_seed=self.config.permutation_seed,
            )
        else:
            if self.capture is None:
                raise RuntimeError("semantic mode was created without a Q/K capture")
            query, key = self.capture.get_conditional(
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                batch_size=batch,
            )
            grid_height, grid_width = infer_token_grid(query.shape[1], height, width)
            graph, entropy = affinity_from_qk(
                query,
                key,
                attention_module=self.attention_module,
                mode=self.config.mode,
                topk=self.config.topk,
                permutation_seed=self.config.permutation_seed,
            )
            coarse_height, coarse_width = grid_height, grid_width

        confidence = (1.0 - entropy).clamp(0.0, 1.0)
        guided = fixed_moment_transport(
            x0,
            graph,
            angle=self.config.angle,
            confidence=confidence,
            grid_height=coarse_height,
            grid_width=coarse_width,
        )
        update = guided - x0
        update_norm = torch.linalg.vector_norm(update.float().reshape(batch, -1), dim=-1)
        scheduler_norm = torch.linalg.vector_norm(
            scheduler_update.float().reshape(batch, -1), dim=-1
        )
        ratio = update_norm / scheduler_norm.clamp_min(torch.finfo(torch.float32).eps)
        self._entropy.extend(entropy.detach().cpu().tolist())
        self._confidence.extend(confidence.detach().cpu().tolist())
        self._ratios.extend(ratio.detach().cpu().tolist())
        self._transport_norms.extend(update_norm.detach().cpu().tolist())
        self._scheduler_norms.extend(scheduler_norm.detach().cpu().tolist())
        return inject_predicted_clean_update(prev_sample, x0, guided)

    def diagnostics(self) -> Dict[str, object]:
        def mean(values):
            return float(sum(values) / len(values)) if values else None

        return {
            "semantic_transport_mode": self.config.mode,
            "semantic_transport_layer": self.config.layer_name,
            "semantic_transport_angle": float(self.config.angle),
            "semantic_transport_topk": int(self.config.topk),
            "semantic_transport_steps": len(self._ratios),
            "mean_normalized_affinity_entropy": mean(self._entropy),
            "mean_transport_confidence": mean(self._confidence),
            "mean_transport_scheduler_update_norm_ratio": mean(self._ratios),
            "mean_transport_update_norm": mean(self._transport_norms),
            "mean_scheduler_update_norm": mean(self._scheduler_norms),
        }
