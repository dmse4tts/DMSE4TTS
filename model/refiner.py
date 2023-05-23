""" Adapted from https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS/model/tts.py
Incorporated with the dpm solver."""

# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.


import random
import torch

from model.base import BaseModule
from model.diffusion import Diffusion
from model.utils import sequence_mask, fix_len_compatibility


class DiffRefiner(BaseModule):
    def __init__(self, n_feats, use_text, dec_dim, beta_min, beta_max, pe_scale):
        super(DiffRefiner, self).__init__()
        self.n_feats = n_feats
        self.use_text = use_text
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        self.decoder = Diffusion(dec_dim, n_feats, use_text, beta_min, beta_max, pe_scale)

    @torch.no_grad()
    def forward(self, y, y_lengths, n_timesteps, temperature=1.0, stoc=False, mu=None):
        """
        Enhance mel-spectrograms.
        
        Args:
            y (torch.Tensor): batch of mel-spectrograms, padded.
            y_lengths (torch.Tensor): length of mel-spectrograms in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): deprecated argument when using the dpm solver.
            mu (torch.Tensor): average mel-spectrogram corresponding to the text. only used for DMSEtext.
        """
        y, y_lengths = self.relocate_input([y, y_lengths])

        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(y.dtype)
        if y_max_length < y_max_length_:
            y = torch.cat((y, y.new(y.shape[0], y.shape[1], y_max_length_-y_max_length).zero_()), dim=2)
            if not isinstance(mu, type(None)):
                mu = torch.cat(
                    (mu, mu.new(mu.shape[0], mu.shape[1], y_max_length_-y_max_length).zero_()), dim=2)

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = torch.randn_like(y, device=y.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, y_mask, y, n_timesteps, stoc, mu)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return decoder_outputs


    def compute_loss(self, x, y, y_lengths, mu=None, out_size=None):
        """
        Compute the diffusion loss
            
        Args:
            x (torch.Tensor): batch of clean mel-spectrograms.
            y (torch.Tensor): batch of the corresponding degraded mel-spectrograms.
            y_lengths (torch.Tensor): length of mel-spectrograms in batch.
            mu (torch.Tensor): average mel-spectrogram corresponding to the text. only used for DMSEtext.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        x, y, y_lengths = self.relocate_input([x, y, y_lengths])

        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(y)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor([
                torch.tensor(random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(y_lengths)
            
            y_cut = torch.zeros(
                y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)
            x_cut = torch.zeros(
                x.shape[0], self.n_feats, out_size, dtype=x.dtype, device=x.device)
            if not isinstance(mu, type(None)):
                mu_cut = torch.zeros(
                    mu.shape[0], self.n_feats, out_size, dtype=mu.dtype, device=mu.device)
            else:
                mu_cut = None

            y_cut_lengths = []
            for i, (y_, x_, out_offset_) in enumerate(zip(y, x, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                x_cut[i, :, :y_cut_length] = x_[:, cut_lower:cut_upper]
                if not isinstance(mu, type(None)):
                    mu_cut[i, :, :y_cut_length] = mu[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths, max_length=out_size).unsqueeze(1).to(y_mask)
            
            y = y_cut
            x = x_cut
            mu = mu_cut
            y_mask = y_cut_mask

        # Compute loss of score-based decoder
        diff_loss, _ = self.decoder.compute_loss(x, y_mask, y, mu)

        return diff_loss