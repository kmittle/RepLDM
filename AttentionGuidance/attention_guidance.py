import torch.nn.functional as F
from torch import Tensor
import torch
from torch import fft

import math
from typing import Optional, Sequence, Tuple, Union


Scale = Union[float, Tensor]
BandScales = Union[Sequence[Scale], Tensor]


class AttnGuidance:
    """
    Args:
        num_total_steps: total sampling steps.
        attn_type: only "vanilla" is implemented.
            Default: "vanilla"
        guidance_scale: non-negative float, typically in [0, 1].
            Default: 0.001
        guidance_density: This divides the sampling into several stages. You can give positive decimals in the form of 
            tuples that represent the proportion of guidance steps in each of the stages. The length of the tuple should 
            not exceed the number of sampling steps, and the number of sampling steps can be evenly divided. You can also 
            give the string "all", which means that guidance is used for all time steps.
            NOTE: When given a tuple, the tuple elements from left to right must correspond to the time step from small to
                large.
            Default: "all"
        guidance_scale_decay: Given a tuple with three elements or None. (decay_strategy, min_scale, factor).
            decay_strategy: choose from {'linear', 'cosine', 'exp'}. 
            min_scale: make sure 0 <= min_scale <= guidance_scale.
            factor:
                When 'linear' or 'cosine' has been chosen, make sure factor >= 1.
                When 'exp' has been chosen, make sure 0 <= factor <= 1.
            When None has been given, guidance_scale will keep unchanged.
            Default: None
        power_calibrate: ensure signal's power is unchanged after attention guidance.
            0: no calibration
            1: take both signal's mean and variance into consideration.
            2: only take signal's variance into consideration.
            Default: None
        guidance_filter: Apply filtering to Training-Free-Self-Attention(latents) so that the part used for guidance gradually includes
            more high-frequency signals. Ensure the input is either None or a tuple containing two elements. When set to
            None, no filtering is applied. When set to a tuple, the first parameter is the method of expanding the filter
            window length, which can be chosen from "linear", "cosine", or "exp"; the second parameter represents the
            initial size of the filter window length, selected from the range [0, 1].
            Default: None
        attn_scaling: w = softmax(XX^T/attn_scaling). Float or None. When it is None, using num_channels ** 0.5.
            Default: None
    """
    def __init__(
        self,
        dtype,
        device,
        num_total_steps: int,
        h: int,
        w: int,
        attn_type: str = "vanilla",
        guidance_scale: float = 0.001,
        guidance_density: Union[str, tuple, list] = "all",
        guidance_scale_decay: Union[None, tuple, list] = None,
        power_calibrate: Union[None, int] = None,
        guidance_filter: Union[None, tuple, list] = None,
        attn_scaling: Optional[float] = None,
        frequency_band_cutoffs: Tuple[float, float] = (0.08, 0.25),
    ) -> None:
        assert num_total_steps > 0
        assert attn_type in {"vanilla"}, "attn_type should be 'vanilla' currently."
        assert 0 <= guidance_scale
        if guidance_density != "all":
            if type(guidance_density) is list:
                guidance_density = tuple(guidance_density)
            else:
                assert type(guidance_density) is tuple
            assert len(guidance_density) > 0
            assert num_total_steps % len(guidance_density) == 0
            for i in guidance_density:
                assert 0 <= i <= 1
        if guidance_scale_decay is not None:
            if type(guidance_scale_decay) is list:
                guidance_scale_decay = tuple(guidance_scale_decay)
            else:
                assert type(guidance_scale_decay) is tuple
            assert len(guidance_scale_decay) == 3
            assert guidance_scale_decay[0] in {"linear", "cosine", "exp"}
            assert 0 <= guidance_scale_decay[1] <= guidance_scale
            if guidance_scale_decay[0] in {'linear', 'cosine'}:
                assert type(guidance_scale_decay[2]) in {float, int} and guidance_scale_decay[2] >= 1
            elif guidance_scale_decay[0] == 'exp':
                assert type(guidance_scale_decay[2]) in {float, int} and 0 <= guidance_scale_decay[2] <= 1
        assert power_calibrate in {0, 1, 2, None}
        if guidance_filter is not None:
            if type(guidance_filter) is list:
                guidance_filter = tuple(guidance_filter)
            else:
                assert type(guidance_filter) is tuple
            assert len(guidance_filter) == 2
            assert guidance_filter[0] in {'linear', 'cosine', 'exp'}
            assert 0 <= guidance_filter[1] <= 1
        if attn_scaling is not None: assert attn_scaling > 0
        assert len(frequency_band_cutoffs) == 2
        assert 0 < frequency_band_cutoffs[0] < frequency_band_cutoffs[1] < 0.5
        
        self.dtype = dtype
        self.device = device
        self.h = h
        self.w = w
        self.num_total_steps = num_total_steps
        self.attn_type = attn_type
        self.guidance_scale = guidance_scale
        self.guidance_density = guidance_density
        self.guidance_scale_decay = guidance_scale_decay
        self.power_calibrate = power_calibrate
        self.guidance_filter = guidance_filter
        self.attn_scaling = attn_scaling
        self.frequency_band_cutoffs = frequency_band_cutoffs
        self._frequency_mask_cache = {}
        
        self.guidance_step_index = self.determine_guidance_step_index()
        self._guided_t_indices = tuple(sorted(self.guidance_step_index, reverse=True))
        self._guidance_rank_by_t_index = {
            t_index: rank for rank, t_index in enumerate(self._guided_t_indices)
        }
        self.guidance_step_scale = self.determine_guidance_step_scale()
        self.filter_range = self.determine_filter_range()
        self._controlled_filter_range = self.determine_filter_range(self.num_total_steps)
    
    @torch.no_grad()
    def determine_guidance_step_index(self):
        if self.guidance_density == "all":
            guidance_step_index = {i.item() for i in torch.arange(self.num_total_steps)}
        else:
            num_stages = len(self.guidance_density)
            num_stage_steps = self.num_total_steps // len(self.guidance_density)
            num_stage_guidance_steps = tuple(int(num_stage_steps * density) for density in self.guidance_density)
            stage_interval = tuple(num_stage_steps // i if i >= 1 else -1 for i in num_stage_guidance_steps)
            guidance_step_index = set()
            for stage_index in range(num_stages):
                repeat_times = num_stage_guidance_steps[stage_index]
                interval = stage_interval[stage_index]
                if repeat_times == 0 or interval == -1: continue
                stage_end_index = stage_index * num_stage_steps + num_stage_steps - 1
                for repeat in range(repeat_times):
                    guidance_index = stage_end_index - repeat * interval
                    guidance_step_index.add(guidance_index)
        assert len(guidance_step_index) > 0
        return guidance_step_index
    
    @torch.no_grad()
    def determine_guidance_step_scale(self):
        num_guidance_steps = len(self.guidance_step_index)
        max_scale = self.guidance_scale
        dtype = self.dtype
        device = self.device
        if self.guidance_scale_decay is None:
            step_scale = torch.tensor([max_scale for _ in range(num_guidance_steps)],
                                      dtype=dtype, device=device)
            return step_scale
        
        decay_type = self.guidance_scale_decay[0]
        min_scale = self.guidance_scale_decay[1]
        factor = self.guidance_scale_decay[2]
        if decay_type == 'linear':
            step_scale = max_scale * torch.linspace(1, 0, num_guidance_steps, dtype=dtype, device=device) ** factor
            step_scale[step_scale < min_scale] = min_scale
        elif decay_type == 'cosine':
            omega = torch.linspace(0, torch.pi, num_guidance_steps, dtype=dtype, device=device)
            cos_value = ((torch.cos(omega) + 1) / 2) ** factor
            step_scale = max_scale * cos_value
            step_scale[step_scale < min_scale] = min_scale
        elif decay_type == 'exp':
            rate = torch.tensor([factor ** i for i in range(num_guidance_steps)], dtype=dtype, device=device)
            step_scale = max_scale * rate
            step_scale[step_scale < min_scale] = min_scale
        return step_scale
    
    @torch.no_grad()
    def determine_filter_range(self, num_steps: Optional[int] = None):
        guidance_filter = self.guidance_filter
        if guidance_filter is None:
            filter_range = 'full'
        else:
            num_guidance_steps = num_steps or len(self.guidance_step_index)
            device = self.device
            filter_strategy = guidance_filter[0]
            h = self.h // 2
            w = self.w // 2
            filter_start_h = int(guidance_filter[1] * h)
            filter_start_w = int(guidance_filter[1] * w)
            if filter_strategy == 'linear':
                h_filter_range = torch.linspace(filter_start_h, h, num_guidance_steps, dtype=torch.int, device=device)
                w_filter_range = torch.linspace(filter_start_w, w, num_guidance_steps, dtype=torch.int, device=device)
            elif filter_strategy == 'cosine':
                omega = torch.linspace(-torch.pi, 0, num_guidance_steps, device=device)
                h_filter_range = (torch.cos(omega) + 1) / 2
                w_filter_range = h_filter_range
                h_filter_range = h * h_filter_range
                w_filter_range = w * w_filter_range
                h_filter_range = h_filter_range.type(torch.int)
                w_filter_range = w_filter_range.type(torch.int)
                h_filter_range[h_filter_range < filter_start_h] = filter_start_h
                w_filter_range[w_filter_range < filter_start_w] = filter_start_w
            elif filter_strategy == 'exp':
                h_filter_range = torch.logspace(torch.log(torch.tensor(filter_start_h)), torch.log(torch.tensor(h)),
                                                num_guidance_steps, torch.exp(torch.tensor(1)),
                                                dtype=torch.int, device=device)
                w_filter_range = torch.logspace(torch.log(torch.tensor(filter_start_w)), torch.log(torch.tensor(w)),
                                                num_guidance_steps, torch.exp(torch.tensor(1)),
                                                dtype=torch.int, device=device)
            filter_range = tuple(
                (int(h_threshold), int(w_threshold))
                for h_threshold, w_threshold in zip(h_filter_range, w_filter_range)
            )
        return filter_range
    
    def filter(self, x: Tensor, t_index: int, controlled: bool = False) -> Tensor:
        if self.guidance_filter is not None:
            if not 0 <= t_index < self.num_total_steps:
                raise IndexError(
                    f"t_index must be in [0, {self.num_total_steps}), got {t_index}"
                )
            if controlled:
                filter_rank = self.num_total_steps - 1 - t_index
                ranges = self._controlled_filter_range
            else:
                try:
                    filter_rank = self._guidance_rank_by_t_index[t_index]
                except KeyError as exc:
                    raise ValueError(
                        f"legacy filtering is undefined for unguided t_index={t_index}"
                    ) from exc
                ranges = self.filter_range
            h_threshold, w_threshold = ranges[filter_rank]
            # fft
            dtype = x.dtype
            x = x.type(torch.float32)
            x = fft.fftn(x, dim=(-2, -1))
            x = fft.fftshift(x, dim=(-2, -1))
            # filter
            _, _, H, W = x.shape
            mask = torch.zeros((1, 1, H, W), device=x.device)
            crow, ccol = H // 2, W // 2
            mask[..., crow - h_threshold:crow + h_threshold, ccol - w_threshold:ccol + w_threshold] = 1
            x = x * mask
            # ifft
            x = fft.ifftshift(x, dim=(-2, -1))
            x = fft.ifftn(x, dim=(-2, -1)).real
            x = x.type(dtype)
        return x

    def _frequency_masks(self, height: int, width: int, device: torch.device) -> Tensor:
        """Return smooth low/mid/high masks forming a partition of unity."""
        key = (height, width, str(device))
        cached = self._frequency_mask_cache.get(key)
        if cached is not None:
            return cached

        fy = torch.fft.fftfreq(height, device=device, dtype=torch.float32)
        fx = torch.fft.rfftfreq(width, device=device, dtype=torch.float32)
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
        low_cutoff, mid_cutoff = self.frequency_band_cutoffs
        low_pass = torch.exp(-0.5 * (radius / low_cutoff).pow(4))
        mid_pass = torch.exp(-0.5 * (radius / mid_cutoff).pow(4))
        masks = torch.stack((low_pass, mid_pass - low_pass, 1.0 - mid_pass))
        self._frequency_mask_cache[key] = masks
        return masks

    @staticmethod
    def _coerce_band_scales(band_scales: BandScales, reference: Tensor) -> Tensor:
        if isinstance(band_scales, Tensor):
            scales = band_scales.to(device=reference.device, dtype=torch.float32)
        else:
            values = [
                value.to(device=reference.device, dtype=torch.float32)
                if isinstance(value, Tensor)
                else torch.tensor(value, device=reference.device, dtype=torch.float32)
                for value in band_scales
            ]
            scales = torch.stack(values)
        if scales.ndim not in (1, 2) or scales.shape[-1] != 3:
            raise ValueError(
                "band_scales must have shape (3,) or (batch, 3) for low/mid/high gains"
            )
        if scales.ndim == 2 and scales.shape[0] != reference.shape[0]:
            raise ValueError(
                f"batched band_scales has batch {scales.shape[0]}, expected {reference.shape[0]}"
            )
        return scales

    def apply_frequency_band_scales(self, residual: Tensor, band_scales: BandScales) -> Tensor:
        """Apply low/mid/high gains to a TFSA residual with one inverse FFT."""
        if not isinstance(band_scales, Tensor):
            scalar_values = [value for value in band_scales if not isinstance(value, Tensor)]
            if len(scalar_values) == 3 and scalar_values[0] == scalar_values[1] == scalar_values[2]:
                legacy_scale = torch.tensor(
                    scalar_values[0], device=residual.device, dtype=residual.dtype
                )
                return legacy_scale * residual
        scales = self._coerce_band_scales(band_scales, residual)
        height, width = residual.shape[-2:]
        masks = self._frequency_masks(height, width, residual.device)
        spectrum = fft.rfft2(residual.float(), dim=(-2, -1), norm="ortho")
        if scales.ndim == 1:
            combined_mask = torch.sum(scales[:, None, None] * masks, dim=0)
        else:
            combined_mask = torch.sum(
                scales[:, :, None, None] * masks[None, ...], dim=1
            )[:, None, ...]
        update = fft.irfft2(
            spectrum * combined_mask,
            s=(height, width),
            dim=(-2, -1),
            norm="ortho",
        )
        return update.to(residual.dtype)

    @staticmethod
    def limit_update_ratio(
        update: Tensor,
        reference_update: Tensor,
        max_update_ratio: Scale,
    ) -> Tensor:
        """Cap guidance energy relative to the scheduler update, per sample."""
        if update.shape != reference_update.shape:
            raise ValueError("reference_update must have the same shape as the guidance update")
        ratio = torch.as_tensor(max_update_ratio, device=update.device, dtype=torch.float32)
        if torch.any(ratio < 0):
            raise ValueError("max_update_ratio must be non-negative")
        if ratio.ndim > 1 or (ratio.ndim == 1 and ratio.shape[0] not in (1, update.shape[0])):
            raise ValueError("max_update_ratio must be scalar or have one value per batch item")
        dims = tuple(range(1, update.ndim))
        update_norm = torch.linalg.vector_norm(update.float(), dim=dims)
        reference_norm = torch.linalg.vector_norm(reference_update.float(), dim=dims)
        ratio = ratio.reshape(-1)
        if ratio.numel() == 1:
            ratio = ratio.expand_as(update_norm)
        multiplier = torch.clamp(
            ratio * reference_norm / (update_norm + 1e-12), max=1.0
        )
        return update * multiplier.reshape((-1,) + (1,) * (update.ndim - 1)).to(update.dtype)
    
    def vanilla_attn_guidance(self, latents: Tensor, alpha_t: Optional[Tensor] = None) -> Tensor:
        b, c, h, w = latents.shape
        scaling = c ** 0.5 if self.attn_scaling is None else self.attn_scaling
        latents = latents.reshape(b, c, -1)
        k = latents
        latents = latents.transpose(-1, -2)
        q = latents / scaling
        attn = torch.matmul(q, k)
        attn = F.softmax(attn, dim=-1)
        latents_ = torch.matmul(attn, latents)
        if self.power_calibrate and alpha_t is None:
            raise ValueError("alpha_t is required when power_calibrate is set")
        if self.power_calibrate:
            if self.power_calibrate == 1:
                power = (alpha_t * (latents_ / (latents + 1e-6)) ** 2 + (1 - alpha_t) * (attn ** 2).sum(dim=-1, keepdim=True)) ** 0.5
            else:
                power = (alpha_t + (1 - alpha_t) * (attn ** 2).sum(dim=-1, keepdim=True)) ** 0.5
            latents_ = latents_ / (power + 1e-6)
        latents_ = latents_.transpose(-1, -2).reshape(b, c, h, w)
        return latents_
    
    def __call__(
        self,
        t_index: int,
        latents: Tensor,
        alpha_t: Optional[Tensor] = None,
        scale: Optional[Scale] = None,
        band_scales: Optional[BandScales] = None,
        reference_update: Optional[Tensor] = None,
        max_update_ratio: Optional[Scale] = None,
    ) -> Tensor:
        """
        NOTE: Here, t_index is not the same as timestep because Diffusion Models typically use a skip-step sampling
        strategy. t_index represents the index of the timestep. For a sampling with T=50, t=50, ..., 1 corresponds
        to t_index=49, ..., 0.

        scale: per-step scalar guidance strength.
            - None (default): use the precomputed hand-tuned schedule (gated by guidance_step_index / decayed by
              guidance_scale_decay). The whole op runs under torch.no_grad() -- byte-for-byte the original
              training-free path.
            - float | Tensor (e.g. emitted by a learned controller): overrides the schedule and is applied at EVERY
              step, leaving the op grad-enabled so gradients can flow through `scale` (and, if `latents` requires
              grad, through the attention nudge) for differentiable reward fine-tuning. scale == 0 reproduces the
              no-guidance latent, so a learned controller subsumes guidance_density (0 == OFF) AND guidance_scale_decay.
        band_scales: low/mid/high gains applied to a smooth spectral decomposition
            of the TFSA residual. Equal gains recover scalar guidance.
        max_update_ratio: optional trust-region cap on guidance-update norm relative
            to `reference_update`, normally the scheduler's update at this step.
        """
        if scale is not None and band_scales is not None:
            raise ValueError("provide either scale or band_scales, not both")
        controlled = scale is not None or band_scales is not None
        if not controlled:
            with torch.no_grad():
                if t_index in self.guidance_step_index:
                    rank = self._guidance_rank_by_t_index[t_index]
                    s = self.guidance_step_scale[rank]
                    if self.attn_type == 'vanilla':
                        guided = self.vanilla_attn_guidance(
                            self.filter(latents, t_index, controlled=False), alpha_t
                        )
                        latents = latents + s * (guided - latents)
            return latents

        if self.attn_type == 'vanilla':
            guided = self.vanilla_attn_guidance(
                self.filter(latents, t_index, controlled=True), alpha_t
            )
            residual = guided - latents
            update = (
                scale * residual
                if band_scales is None
                else self.apply_frequency_band_scales(residual, band_scales)
            )
            if max_update_ratio is not None:
                if reference_update is None:
                    raise ValueError("reference_update is required with max_update_ratio")
                update = self.limit_update_ratio(update, reference_update, max_update_ratio)
            latents = latents + update
        return latents


if __name__ == '__main__':
    attn_guidance = AttnGuidance(dtype=torch.float16, device='cpu', num_total_steps=50,
                                      h = 1024, w = 2048,
                                      guidance_scale=3e-3,
                                      guidance_density=tuple([1] * 47 + [0] * 3),
                                      guidance_scale_decay=('linear', 0, 3),
                                      guidance_filter=None,)
    print(attn_guidance.guidance_step_index)
    print(len(attn_guidance.guidance_step_index))
    print([round(value.item(), 4) for value in attn_guidance.guidance_step_scale])
    print(attn_guidance.filter_range)
