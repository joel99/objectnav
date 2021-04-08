#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from typing import Dict, List, Optional

import torch
import torch.nn as nn

ACTION_EMBEDDING_DIM = 4

def subsampled_mean(x, p: float=0.1):
    return torch.masked_select(x, torch.rand_like(x) < p).mean()

def hard_mining_mean(x, p: float=0.1, across_batches: bool=False):
    flat_dim = min(0 if across_batches else 1, len(x.size()) - 1)
    x_flat = x.flatten(start_dim=flat_dim)
    loss_sort = torch.argsort(x_flat, dim=flat_dim, descending=True) # batch being first dim
    sorted_by_hard = x_flat[loss_sort]
    end_index = int(sorted_by_hard.size(flat_dim) * p)
    if flat_dim == 0:
        return sorted_by_hard[:end_index].mean()
    return sorted_by_hard[:, :end_index].mean()

SUBSAMPLER = {
    "random": subsampled_mean,
    "hard": hard_mining_mean
}

class RolloutAuxTask(nn.Module):
    r""" Rollout-based self-supervised auxiliary task base class.
    """

    def __init__(self, cfg, aux_cfg, task_cfg, device, hidden_size=None, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.aux_cfg = aux_cfg
        self.task_cfg = task_cfg # Mainly tracked for actions
        self.device = device
        self.aux_hidden_size = hidden_size if hidden_size is not None else cfg.hidden_size
        self.loss_factor = getattr(aux_cfg, "loss_factor", 0.1) # absent for dummy
        self.subsample_rate = getattr(aux_cfg, "subsample_rate", 0.1)
        self.strategy = getattr(aux_cfg, "sample", "random")

    def forward(self):
        raise NotImplementedError

    @torch.jit.export
    @abc.abstractmethod
    def get_loss(self,
        observations: Dict[str, torch.Tensor],
        actions,
        sensor_embeddings: Dict[str, torch.Tensor],
        final_belief_state,
        belief_features,
        metrics: Dict[str, torch.Tensor],
        n: int,
        t: int,
        env_zeros: List[List[int]]
    ):
        pass

    @staticmethod
    def get_required_sensors(*args):
        # Encoders required to function (not habitat sensors, those are specified elsewhere)
        return []

    def masked_sample_and_scale(self, x, mask: Optional[torch.Tensor]=None):
        if mask is not None:
            x = torch.masked_select(x, mask)
        # strat_name = getattr(cfg, "sample", "random")
        # sampler = SUBSAMPLER.get(self.strategy, "random")
        # Drop support for hard mining
        return subsampled_mean(x, p=self.subsample_rate) * self.loss_factor