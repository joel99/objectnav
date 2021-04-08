#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as distrib
import torch.nn as nn

# Changes reflect : https://github.com/facebookresearch/habitat-lab/compare/etw-updates...fp16

@torch.jit.script
def welford_update(mean, var, count, new_mean, new_var, new_count):
    m_a = var * count
    m_b = new_var * new_count
    new_count_total = count + new_count

    M2 = m_a + m_b + (new_mean - mean).pow(2) * count * new_count / new_count_total

    var = M2 / new_count_total
    mean = (count * mean + new_count * new_mean) / new_count_total

    return var, mean, new_count_total


# @torch.jit.script
def apply_mean_var(x, mean, var, eps):
    inv_stdev = torch.rsqrt(torch.max(var, eps))

    return torch.addcmul(
        (-mean.type_as(x) * inv_stdev.type_as(x)), inv_stdev.type_as(x), x, # value=1.0
    )


@torch.jit.script
def inv_apply_mean_var(x, mean, var, eps):
    stdev = torch.sqrt(torch.max(var, eps))

    return torch.addcmul(mean.to(x.dtype), stdev.to(x.dtype), x, value=1.0)

class RunningMeanAndVar(nn.Module):
    def __init__(self, n_channels=None, shape=None, eps=1e-2, initial_count=1e-2):
        super().__init__()
        assert n_channels is None or shape is None
        if n_channels is not None:
            shape = (1, n_channels, 1, 1)

        self.register_buffer("_mean", torch.zeros(shape))
        self.register_buffer("_var", torch.ones(shape))
        self.register_buffer("_count", torch.full((), initial_count))
        self.register_buffer("_eps", torch.full((), eps))

        self._distributed = distrib.is_initialized()
        self._shape = shape

    def normalize(self, x):
        return apply_mean_var(x, self._mean, self._var, self._eps)

    def denormalize(self, x):
        return inv_apply_mean_var(x, self._mean, self._var, self._eps)

    @property
    def dtype(self):
        return self._mean.dtype

    def update(self, x):
        r""" I think this is supposed to come in as B x C x H x W"""
        with torch.no_grad():
            x_channel_first = x.to(self.dtype).transpose(0, 1).reshape(x.size(1), -1)
            new_count = torch.full_like(self._count, x.size(0)).float()
            new_mean = x_channel_first.mean(-1)

            if self._distributed:
                distrib.all_reduce(new_count)

                new_mean = new_mean.float()
                distrib.all_reduce(new_mean)

                # msg = torch.cat([new_mean, new_count.unsqueeze(-1)])
                # distrib.all_reduce(msg)
                # new_mean = msg[0:-1]
                # new_count = msg[-1]

                new_mean /= distrib.get_world_size()

            # new_var = (x_channel_first - new_mean.view(x.size(1), -1)).pow(2).mean(-1)
            new_var = (
                x_channel_first - new_mean.view(x.size(1), -1).type_as(x)
            ).pow(2).mean(-1, keepdim=True).float()

            if self._distributed:
                distrib.all_reduce(new_var)
                new_var /= distrib.get_world_size()
            new_var = new_var.view(self._shape)
            new_mean = new_mean.view(self._shape).float()

            self._mean, self._var, self._count = welford_update(
                self._mean, self._var, self._count, new_mean, new_var, new_count,
            )

    def forward(self, x):
        if self.training:
            self.update(x)

        return self.normalize(x)