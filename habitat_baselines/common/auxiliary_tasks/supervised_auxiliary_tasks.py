#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from habitat import logger
from habitat.tasks.nav.object_nav_task import (
    task_cat2mpcat40
)

from habitat_baselines.common.utils import Flatten
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.auxiliary_tasks.aux_utils import (
    ACTION_EMBEDDING_DIM, RolloutAuxTask
)
from habitat_baselines.common.auxiliary_tasks.auxiliary_tasks import CPCA
import habitat_baselines.rl.models.resnet as resnet

r"""
    Supervised tasks are distinguished by their access to ground truth sensors (or environmental metrics) unavailable at test time. We mainly refer to the semantic sensor, here.
    Though, we plan on adding maskrcnn as a dropin for ground truth semantics. As such, we could probably still fine-tune at test.

    TODO: Implementation-wise, supervised tasks will request the attachment of their supervised sensors to the policy.
"""

# ======================
# * New tasks begin
# ======================

@baseline_registry.register_aux_task(name="SemanticGoalExists")
class SemanticGoalExists(RolloutAuxTask):
    r"""
        Force belief to decode whether goal exists. This is likely going to forward to the vision. We use it
        - in the event that recurrent processing helps a little
        - in hopes it forces the agent to understand where the flag calculation is coming from.
        On the flip side, this task is noisier than just using the flag.
        - This requires test-time semantic sensors. This may not be a problem, but we're not there yet.
        - We should compare the two variants. I can't intuit which is better.
    """

    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        self.decoder = nn.Linear(self.aux_hidden_size, 1)
        self.task_cat2mpcat40 = torch.tensor(task_cat2mpcat40, dtype=torch.long, device=device)

    @staticmethod
    def get_required_sensors(*args):
        return [] # this requires the sensor, but not the encoder. Sensor requirements are specified in overall config

    @torch.jit.export
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
        semantic_obs = observations["semantic"].long()
        semantic_obs = semantic_obs.flatten(start_dim=1)

        # goal_category = self.task_cat2mpcat40[observations["objectgoal"].long()]
        # pixel_threshold = semantic_obs.size(-1) * self.aux_cfg.threshold
        # matches = (semantic_obs == goal_category).sum(dim=1) > pixel_threshold # b x pixels -> b
        matches = metrics["goal_vis"]
        match_logits = self.decoder(belief_features)

        loss = F.mse_loss(
            match_logits.flatten(), matches.float().flatten(), reduction='none'
        )

        return self.masked_sample_and_scale(loss)

# ! Measures tend to be recorded for our benefit, but we might as well give them to the agent at training time
# ! unfortunately measures aren't batched, so they might be slower. Thus, we'll make a separate task that pulls from metric, otherwise we won't at all.
@baseline_registry.register_aux_task(name="CoveragePrediction")
class DeltaPrediction(RolloutAuxTask):
    r"""
        Decode coverage metrics as exploration progress estimation.
        A belief that successfully encodes coverage information and can predict how it changes must encode an internal map that tracks history and unexplored spaces. This would be useful for exploration.
        Specifically, conditioned on:
        - belief
        - actions
        Decode
        - delta in coverage between consecutive steps
        - maybe this should be CEL instead?
        Note: It's tricky to figure out exactly when the delta occurs.
        Other possibilities include:
        - DTW
        - Label smoothing + BCE (not quite satisfying)
        - MSELoss with cumsum. (we're going with this)
        To somewhat help with the tile edge problem, I'll feed in the visit count for the first tile.
        Architecture based off of CPCA.
    """

    def __init__(self, cfg, aux_cfg, task_cfg, device, observation_space=None, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        num_actions = len(task_cfg.POSSIBLE_ACTIONS)
        self.regression = aux_cfg.regression
        if self.regression:
            self.classifier = nn.Linear(aux_cfg.hidden_size, 1)
        else:
            self.classifier = nn.Linear(aux_cfg.hidden_size, 10)
        self.action_embedder = nn.Embedding(num_actions + 1, ACTION_EMBEDDING_DIM)
        self.state_projector = nn.Linear(self.aux_hidden_size + 1, aux_cfg.hidden_size)
        self.query_gru = nn.GRU(ACTION_EMBEDDING_DIM, aux_cfg.hidden_size)
        self.key = aux_cfg.key
        self.k = self.aux_cfg.num_steps

        self.VISIT_COUNT_NORMALIZATION = 500 # Just to get this into the right scale
        self.NUM_BINS = 9 # Inclusive max range. Higher values clamped

    @torch.jit.export
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
        coverage = metrics[self.key] # t x n

        belief_features = belief_features.view(t*n, -1)
        query_in = torch.cat([belief_features, metrics['visit_count'].view(t * n, 1) / self.VISIT_COUNT_NORMALIZATION], -1)
        query_in = self.state_projector(query_in)

        action_embedding = self.action_embedder(actions) # t n -1
        action_padding = torch.zeros(self.k - 1, n, action_embedding.size(-1), device=self.device)
        action_padded = torch.cat((action_embedding, action_padding), dim=0) # (t+k-1) x n x -1
        # t x n x -1 x k
        action_seq = action_padded.unfold(dimension=0, size=self.k, step=1).permute(3, 0, 1, 2)\
            .view(self.k, t*n, action_embedding.size(-1))
        # for each timestep, predict up to k ahead -> (t,k) is query for t + k (+ 1) (since k is 0-indexed) i.e. (0, 0) -> query for 1
        out_all, _ = self.query_gru(action_seq, query_in.unsqueeze(0))
        query_all = out_all.view(self.k, t, n, -1).permute(1, 0, 2, 3) # t x k x n -1
        delta_preds = self.classifier(query_all)[:-1] # Cut off the last step, it's entirely padding (should ideally be cut earlier, but not messing with that right now)
        # Targets: predict k steps for each starting timestep
        coverage_padded = torch.cat((coverage[1:], torch.zeros(self.k, n, dtype=torch.long, device=self.device)), dim=0) # (t+k) x n
        # Offset by 1 because our predictions take the belief at the end of timestep t
        coverage_at_t_plus_1 = coverage_padded.unfold(dimension=0, size=self.k, step=1).permute(0, 2, 1) # t x k x n
        # coverage_at_t_plus_1[t] describes a slice of k-steps starting at time t+1
        # to get deltas, subtract coverage[t+1]
        delta_at_t = coverage_at_t_plus_1[:-1] - coverage[1:].unsqueeze(1) # t-1 x k x n
        delta_true = delta_at_t.unsqueeze(-1)

        # Masking - we mask out episode crossing
        # Note which timesteps [1, t+k+1] could have valid queries, at distance (k) (note offset by 1)
        valid_modeling_queries = torch.ones(
            t + self.k, self.k, n, 1, device=self.device, dtype=torch.bool
        ) # (padded) timestep predicted x prediction distance x env (this logic is moved to be from timestep sourced in the diagonalization)
        valid_modeling_queries[t - 1:] = False # timesteps >= t is predicting entirely past rollout
        for j in range(1, self.k + 1): # for j-step predictions
            valid_modeling_queries[:j - 1, j - 1] = False # first j frames cannot be valid for all envs (rollout doesn't go that early)
            for env in range(n):
                has_zeros_batch = env_zeros[env]
                # in j-step prediction, timesteps z -> z + j are disallowed as those are the first j timesteps of a new episode
                # z-> z-1 because of modeling_queries being offset by 1
                for z in has_zeros_batch:
                    valid_modeling_queries[z-1: z-1 + j, j - 1, env] = False

        # instead of the whole range, we actually are only comparing a window i:i+k for each query/target i - for each, select the appropriate k
        # we essentially gather diagonals from this full mask, t of them, k long such that each diagonal corresponds to one starting timestep
        valid_diagonals = [torch.diagonal(valid_modeling_queries, offset=-i) for i in range(t)] # pull the appropriate k per timestep
        valid_mask = torch.stack(valid_diagonals, dim=0).permute(0, 3, 1, 2)[:-1] # t x n x 1 x k -> (t-1) x k x n x 1

        delta_true = torch.masked_select(delta_true, valid_mask)
        if self.regression:
            delta_preds = torch.masked_select(delta_preds, valid_mask)
            loss = F.mse_loss(
                delta_preds, delta_true.float(), reduction='none'
            )
        else:
            # [t, k, n, c] -> [t * k * n, c]
            delta_preds = delta_preds.flatten(end_dim=2)
            valid_mask_flat = valid_mask.flatten() # [t * k * n]
            delta_preds = delta_preds[valid_mask_flat]
            # TODO We'll add soft labels if this is hard?
            loss = F.cross_entropy(
                delta_preds,
                torch.clamp(delta_true.long(), 0, self.NUM_BINS),
                reduction='none'
            )

        return self.masked_sample_and_scale(loss)

# ======================
# * New tasks end
# ======================