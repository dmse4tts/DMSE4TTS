"""Adapted from https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS/model/diffusion.py Incorporated with the dpm solver. """

# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import torch
from einops import rearrange

from model.base import BaseModule
from model.dpm_solver import model_wrapper, NoiseScheduleVP, DPM_Solver


def normalize_batch(x, seq_len):
    x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
                         device=x.device)
    x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
                        device=x.device)
    for i in range(x.shape[0]):
        x_mean[i, :] = x[i, :, :seq_len[i]].mean(dim=1)
        x_std[i, :] = x[i, :, :seq_len[i]].std(dim=1)
    # make sure x_std is not zero
    x_std += 1e-5
    return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)

class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(BaseModule):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Rezero(BaseModule):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, 3, 
                                         padding=1), torch.nn.GroupNorm(
                                         groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                               dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(BaseModule):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)            

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                            heads = self.heads, qkv=3)            
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Residual(BaseModule):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GradLogPEstimator2d(BaseModule):
    def __init__(self, dim, dim_mults=(1, 2, 4, 8, 8), groups=8,
                 use_text=False, pe_scale=1000):
        super(GradLogPEstimator2d, self).__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.use_text = use_text
        self.pe_scale = pe_scale
        
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim * 4), Mish(),
                                       torch.nn.Linear(dim * 4, dim))

        dims = [2 + (1 if use_text else 0), *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       ResnetBlock(dim_in, dim_out, time_emb_dim=dim, groups=self.groups),
                       ResnetBlock(dim_out, dim_out, time_emb_dim=dim, groups=self.groups),
                       Residual(Rezero(LinearAttention(dim_out))),
                       Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=self.groups)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=dim, groups=self.groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in,
                            time_emb_dim=dim, groups=self.groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim,
                            groups=self.groups),
                     Residual(Rezero(LinearAttention(dim_in))),
                     Upsample(dim_in)]))
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, x, t, mask=None, y=None, mu=None):
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        if (not self.use_text) and isinstance(mu, type(None)):
            x = torch.stack([y, x], 1)
        else:
            x = torch.stack([y, x, mu], 1)
        mask = mask.unsqueeze(1)

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)


def get_noise(t, beta_init, beta_term, cumulative=False):
    if cumulative:
        noise = beta_init*t + 0.5*(beta_term - beta_init)*(t**2)
    else:
        noise = beta_init + (beta_term - beta_init)*t
    return noise


class Diffusion(BaseModule):
    def __init__(self, dim, n_feats=128, use_text=False,
                 beta_min=0.05, beta_max=20, pe_scale=1000):
        super(Diffusion, self).__init__()
        self.dim = dim
        self.n_feats = n_feats
        self.use_text = use_text
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        self.estimator = GradLogPEstimator2d(dim, use_text=use_text,pe_scale=pe_scale)
        self.noise_schedule = NoiseScheduleVP(
            schedule='linear', continuous_beta_0=beta_min, continuous_beta_1=beta_max)

    def forward_diffusion(self, x0, mask, t, mu=None):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        # if not isinstance(mu, type(None)):
        #     mean = x0*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        # else:
        #     mean = x0*torch.exp(-0.5*cum_noise)

        mean = x0*torch.exp(-0.5*cum_noise)
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, y, n_timesteps, stoc=False, mu=None):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max, 
                                cumulative=False)
            if stoc:  # adds stochastic term
                dxt_det = 0.5 * (-xt) - self.estimator(xt, t, mask, y, mu)
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (-xt - self.estimator(xt, t, mask, y, mu))
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
            # xt.clamp_(-1.2, 1.2)
        return xt

    # !!! incorporate with the dpm solver
    @torch.no_grad()
    def reverse_diffusion_dpm(self, z, mask, y, n_timesteps, mu=None):
        xt = z * mask
        model_fn = model_wrapper(
            self.estimator,
            self.noise_schedule,
            model_type='score',
            model_kwargs={'mask': mask, 'y': y, 'mu': mu},
        )
        dpm_solver = DPM_Solver(
            model_fn, self.noise_schedule, algorithm_type='dpmsolver')

        xt = dpm_solver.sample(
            xt,
            steps=n_timesteps,
            order=3,
            skip_type='time_uniform',
            method='singlestep',
        )
        xt = xt * mask
        return xt

    @torch.no_grad()
    def forward(self, z, mask, y, n_timesteps, stoc=False, mu=None):
        # return self.reverse_diffusion(z, mask, y, n_timesteps, stoc, mu)
        return self.reverse_diffusion_dpm(z, mask, y, n_timesteps, mu)

    def loss_t(self, x0, mask, y, t, mu=None):
        xt, z = self.forward_diffusion(x0, mask, t, mu)
        # xt, z = self.forward_diffusion(x0, mask, t, y)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        noise_estimation = self.estimator(xt, t, mask, y, mu)
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.sum((noise_estimation + z)**2) / (torch.sum(mask)*self.n_feats)
        return loss, xt

    def compute_loss(self, x0, mask, y, mu=None, offset=1e-5):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, y, t, mu)
