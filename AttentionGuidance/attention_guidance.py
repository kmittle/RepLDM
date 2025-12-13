import torch.nn.functional as F
from torch import Tensor
import torch
from torch import fft

import math
from typing import Union


class AttnGuidance:
    """
    Args:
        num_total_steps: total sampling steps.
        attn_type: choose from {"vanilla", "linear", "swin", "skipwin"}.
            Default: "vanilla"
        guidance_scale: choose from [0, 1].
            Defualt: 0.001
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
            Default: 0/None
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
        w:int,
        attn_type: str = "vanilla",
        guidance_scale: float = 0.001,
        guidance_density: Union[str, tuple, list] = "all",
        guidance_scale_decay: Union[None, tuple, list] = None,
        power_calibrate: float = None,
        guidance_filter: Union[None, tuple, list] = None,
        attn_scaling: float = None,
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
        
        self.guidance_step_index = self.determine_guidance_step_index()
        self.guidance_step_scale = iter(self.determine_guidance_step_scale())
        self.filter_range = self.determine_filter_range()
    
    @ torch.no_grad()
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
    
    @ torch.no_grad()
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
            omega = torch.linspace(0, torch.pi, num_guidance_steps)
            cos_value = ((torch.cos(omega) + 1) / 2) ** factor
            step_scale = max_scale * cos_value
            step_scale[step_scale < min_scale] = min_scale
            step_scale = step_scale.type(dtype)
            step_scale = step_scale.to(device)
        elif decay_type == 'exp':
            rate = torch.tensor([factor ** i for i in range(num_guidance_steps)])
            step_scale = max_scale * rate
            step_scale[step_scale < min_scale] = min_scale
            step_scale = step_scale.type(dtype)
            step_scale = step_scale.to(device)
        return step_scale
    
    @ torch.no_grad()
    def determine_filter_range(self):
        guidance_filter = self.guidance_filter
        if guidance_filter is None:
            filter_range = 'full'
        else:
            num_guidance_steps = len(self.guidance_step_index)
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
                omega = torch.linspace(-torch.pi, 0, num_guidance_steps)
                h_filter_range = (torch.cos(omega) + 1) / 2
                w_filter_range = h_filter_range
                h_filter_range = h * h_filter_range
                w_filter_range = w * w_filter_range
                h_filter_range = h_filter_range.type(torch.int)
                w_filter_range = w_filter_range.type(torch.int)
                h_filter_range[h_filter_range < filter_start_h] = filter_start_h
                w_filter_range[w_filter_range < filter_start_w] = filter_start_w
                h_filter_range = h_filter_range.to(device)
                w_filter_range = w_filter_range.to(device)
            elif filter_strategy == 'exp':
                h_filter_range = torch.logspace(torch.log(torch.tensor(filter_start_h)), torch.log(torch.tensor(h)),
                                                num_guidance_steps, torch.exp(torch.tensor(1)),
                                                dtype=torch.int, device=device)
                w_filter_range = torch.logspace(torch.log(torch.tensor(filter_start_w)), torch.log(torch.tensor(w)),
                                                num_guidance_steps, torch.exp(torch.tensor(1)),
                                                dtype=torch.int, device=device)
            filter_range = tuple(zip(h_filter_range, w_filter_range))
            filter_range = iter(filter_range)
        return filter_range
    
    @ torch.no_grad()
    def filter(self, x):
        if self.guidance_filter is not None:
            h_threshold, w_threshold = next(self.filter_range)
            # fft
            dtype = x.dtype
            x = x.type(torch.float32)
            x = fft.fftn(x, dim=(-2, -1))
            x = fft.fftshift(x, dim=(-2, -1))
            # filter
            B, C, H, W = x.shape
            mask = torch.zeros((B, C, H, W), device=self.device)
            crow, ccol = H // 2, W //2
            mask[..., crow - h_threshold:crow + h_threshold, ccol - w_threshold:ccol + w_threshold] = 1
            x = x * mask
            # ifft
            x = fft.ifftshift(x, dim=(-2, -1))
            x = fft.ifftn(x, dim=(-2, -1)).real
            x = x.type(dtype)
        return x
    
    @ torch.no_grad()
    def vanilla_attn_guidance(self, latents: Tensor, alpha_t: Tensor = None) -> Tensor:
        b, c, h, w = latents.shape
        scaling = c ** 0.5 if self.attn_scaling is None else self.attn_scaling
        latents = latents.reshape(b, c, -1)
        k = latents
        latents = latents.transpose(-1, -2)
        q = latents / scaling
        attn = torch.matmul(q, k)
        attn = F.softmax(attn, dim=-1)
        latents_ = torch.matmul(attn, latents)
        if self.power_calibrate:
            if self.power_calibrate == 1:
                power = (alpha_t * (latents_ / (latents + 1e-6)) ** 2 + (1 - alpha_t) * (attn ** 2).sum(dim=-1, keepdim=True)) ** 0.5
            else:
                power = (alpha_t + (1 - alpha_t) * (attn ** 2).sum(dim=-1, keepdim=True)) ** 0.5
            latents_ = latents_ / (power + 1e-6)
        latents_ = latents_.transpose(-1, -2).reshape(b, c, h, w)
        return latents_
    
    @ torch.no_grad()
    def __call__(self, t_index: int, latents: Tensor, alpha_t: Tensor = None) -> Tensor:
        """
        NOTE: Here, t_index is not the same as timestep because Diffusion Models typically use a skip-step sampling
        strategy. t_index represents the index of the timestep. For a sampling with T=50, t=50, ..., 1 corresponds
        to t_index=49, ..., 0.
        """
        if t_index in self.guidance_step_index:
            scale = next(self.guidance_step_scale)
            if self.attn_type == 'vanilla':
                latents = latents + scale * (self.vanilla_attn_guidance(self.filter(latents), alpha_t) - latents)
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
    print([round(next(attn_guidance.guidance_step_scale).item(), 4) for _ in range(len(attn_guidance.guidance_step_index))])
    print(attn_guidance.filter_range)

