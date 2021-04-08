#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Categorical

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.auxiliary_tasks.aux_utils import (
    ACTION_EMBEDDING_DIM, RolloutAuxTask
)

@baseline_registry.register_aux_task(name="InverseDynamicsTask")
class InverseDynamicsTask(RolloutAuxTask):
    r""" Predict action used between two consecutive frames
    """

    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        num_actions = len(task_cfg.POSSIBLE_ACTIONS)
        self.decoder = nn.Linear(2 * cfg.hidden_size + self.aux_hidden_size, num_actions)

        self.classifier = nn.CrossEntropyLoss(reduction='none')

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
        vision = sensor_embeddings["all"]
        actions = actions[:-1] # t-1 x n
        final_belief_expanded = final_belief_state.expand(t-1, -1, -1) # 1 x n x -1 -> t-1 x n x -1
        decoder_in = torch.cat((vision[:-1], vision[1:], final_belief_expanded), dim=2)
        preds = self.decoder(decoder_in).permute(0, 2, 1) # t-1 x n x 4 -> t-1 x 4 x n
        loss = self.classifier(preds, actions)

        last_zero = torch.zeros(n, dtype=torch.long)
        for env in range(n):
            has_zeros_batch = env_zeros[env]
            if len(has_zeros_batch) == 0:
                has_zeros_batch = [-1]

            last_zero[env] = has_zeros_batch[-1]

        # select the losses coming from valid (same episode) frames
        valid_losses = []
        for env in range(n):
            if last_zero[env] >= t - 3:
                continue
            valid_losses.append(loss[last_zero[env]+1:, env]) # variable length (m,) tensors
        if len(valid_losses) == 0:
            valid_losses = torch.zeros(1, device=self.device, dtype=torch.float)
        else:
            valid_losses = torch.cat(valid_losses) # (sum m, )
        return self.masked_sample_and_scale(valid_losses)

@baseline_registry.register_aux_task(name="ForwardDynamicsTask")
class ForwardDynamicsTask(RolloutAuxTask):
    r""" Predict next observation embedding given previous embedding and action
    """

    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        num_actions = len(task_cfg.POSSIBLE_ACTIONS)
        self.decoder = nn.Linear(ACTION_EMBEDDING_DIM + cfg.hidden_size + self.aux_hidden_size, cfg.hidden_size)
        self.action_embedder = nn.Embedding(num_actions + 1, ACTION_EMBEDDING_DIM)

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
        vision = sensor_embeddings["all"]
        actions = actions[:-1]
        frame_0 = vision[:-1]
        final_belief_expanded = final_belief_state.expand(t-1, -1, -1) # 1 x n x -1 -> t-1 x n x -1
        action_embedding = self.action_embedder(actions) # t-1 x n x 4
        decoder_in = torch.cat((action_embedding, frame_0, final_belief_expanded), dim=2) # t-1 x n x -1

        preds = self.decoder(decoder_in).view(t-1, n, -1)

        frame_1 = vision[1:]
        pred_error = (frame_1 - preds).float()
        loss = 0.5 * (pred_error).pow(2).sum(2) # t-1 x n

        last_zero = torch.zeros(n, dtype=torch.long)
        for env in range(n):
            has_zeros_batch = env_zeros[env]
            if len(has_zeros_batch) == 0:
                has_zeros_batch = [-1]

            last_zero[env] = has_zeros_batch[-1]

        # select the losses coming from valid (same episode) frames
        valid_losses = []
        for env in range(n):
            if last_zero[env] >= t - 3:
                continue
            valid_losses.append(loss[last_zero[env]+1:, env]) # variable length (m,) tensors
        if len(valid_losses) == 0:
            valid_losses = torch.zeros(1, device=self.device, dtype=torch.float)
        else:
            valid_losses = torch.cat(valid_losses) # (sum m, )
        return self.masked_sample_and_scale(valid_losses)

@baseline_registry.register_aux_task(name="TemporalDistanceTask")
class TemporalDistanceAuxTask(RolloutAuxTask):
    r""" Class for calculating timesteps between two randomly selected observations
        Specifically, randomly select `num_pairs` frames per env and predict the frames elapsed
    """

    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        self.decoder = nn.Linear(2 * cfg.hidden_size + self.aux_hidden_size, 1) # 2 * embedding + belief
        self.k = self.aux_cfg.num_pairs

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
        vision = sensor_embeddings["all"]
        final_belief_expanded = final_belief_state.expand(self.k, -1, -1) # 1 x n x -1 -> t-1 x n x -1

        indices = torch.zeros((2, self.k, n), device=self.device, dtype=torch.long)
        trial_normalizer = torch.ones(n, device=self.device, dtype=torch.float)
        # find last zero index to find start of current rollout
        for env in range(n):
            has_zeros_batch = env_zeros[env]
            if len(has_zeros_batch) == 0:
                has_zeros_batch = [-0]

            last_index = has_zeros_batch[-1]
            indices[..., env] = torch.randint(last_index, t, (2, self.k))
            if last_index >= t - 5: # too close, drop the trial
                trial_normalizer[env] = 0
            else:
                trial_normalizer[env] = 1.0 / float(t - last_index) # doesn't cast without for some reason
        frames = [vision[indices.view(2 * k, n)[:, i], i] for i in range(n)]
        frames = torch.stack(frames, dim=1).view(2, k, n, -1)
        decoder_in = torch.cat((frames[0], frames[1], final_belief_expanded), dim=-1)
        pred_frame_diff = self.decoder(decoder_in).squeeze(-1) # output k x n
        true_frame_diff = (indices[1] - indices[0]).float() # k x n
        pred_error = (pred_frame_diff - true_frame_diff) * trial_normalizer.view(1, n)
        loss = 0.5 * (pred_error).pow(2)
        avg_loss = loss.mean()
        return avg_loss * self.loss_factor

@baseline_registry.register_aux_task(name="CPCA")
class CPCA(RolloutAuxTask):
    """ Action-conditional CPC - up to k timestep prediction
        From: https://arxiv.org/abs/1811.06407
    """
    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        num_actions = len(task_cfg.POSSIBLE_ACTIONS)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_size + self.aux_hidden_size, 32), # Frame + belief
            nn.ReLU(),
            nn.Linear(32, 1)
        ) # query and perception
        self.action_dim = ACTION_EMBEDDING_DIM
        self.action_embedder = nn.Embedding(num_actions + 1, self.action_dim)
        self.query_gru = nn.GRU(self.action_dim, self.aux_hidden_size)
        self.dropout = nn.Dropout(aux_cfg.dropout)
        self.k = self.aux_cfg.num_steps

    def get_positives(self, observations: Dict[str, torch.Tensor], sensor_embeddings: Dict[str, torch.Tensor]):
        return sensor_embeddings["all"]

    def get_negatives(self, positives, t: int, n: int):
        negative_inds = torch.randperm(t * n, device=self.device, dtype=torch.int64)
        return torch.gather(
            positives.view(t * n, -1),
            dim=0,
            index=negative_inds.view(t * n, 1).expand(t * n, positives.size(-1)),
        ).view(t, n, -1)

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
        k = self.k # up to t

        belief_features = belief_features.view(t*n, -1).unsqueeze(0)
        positives = self.get_positives(observations, sensor_embeddings)
        negatives = self.get_negatives(positives, t, n)
        positives = self.dropout(positives)
        negatives = self.dropout(negatives)
        action_embedding = self.action_embedder(actions) # t n -1
        action_padding = torch.zeros(k - 1, n, action_embedding
        .size(-1), device=self.device)
        action_padded = torch.cat((action_embedding, action_padding), dim=0) # (t+k-1) x n x -1
        # t x n x -1 x k
        action_seq = action_padded.unfold(dimension=0, size=k, step=1).permute(3, 0, 1, 2)\
            .view(k, t*n, action_embedding.size(-1))

        # for each timestep, predict up to k ahead -> (t,k) is query for t + k (+ 1) (since k is 0-indexed) i.e. (0, 0) -> query for 1
        out_all, _ = self.query_gru(action_seq, belief_features)
        query_all = out_all.view(k, t, n, -1).permute(1, 0, 2, 3)

        # Targets: predict k steps for each starting timestep
        positives_padded = torch.cat((positives[1:], torch.zeros(k, n, positives.size(-1), device=self.device)), dim=0) # (t+k) x n
        positives_expanded = positives_padded.unfold(dimension=0, size=k, step=1).permute(0, 3, 1, 2) # t x k x n x -1
        positives_logits = self.classifier(torch.cat([positives_expanded, query_all], -1))
        negatives_padded = torch.cat((negatives[1:], torch.zeros(k, n, negatives.size(-1), device=self.device)), dim=0) # (t+k) x n x -1
        negatives_expanded = negatives_padded.unfold(dimension=0, size=k, step=1).permute(0, 3, 1, 2) # t x k x n x -1
        negatives_logits = self.classifier(torch.cat([negatives_expanded, query_all], -1))

        # Masking
        # Note which timesteps [1, t+k+1] could have valid queries, at distance (k) (note offset by 1)
        valid_modeling_queries = torch.ones(
            t + k, k, n, 1, device=self.device, dtype=torch.bool
        ) # (padded) timestep predicted x prediction distance x env
        valid_modeling_queries[t - 1:] = False # >= t is past rollout, and t is index t - 1 here
        for j in range(1, k + 1): # for j-step predictions
            valid_modeling_queries[:j - 1, j - 1] = False # first j frames cannot be valid for all envs (rollout doesn't go that early)
            for env in range(n):
                has_zeros_batch = env_zeros[env]
                # in j-step prediction, timesteps z -> z + j are disallowed as those are the first j timesteps of a new episode
                # z-> z-1 because of modeling_queries being offset by 1
                for z in has_zeros_batch:
                    valid_modeling_queries[z-1: z-1 + j, j - 1, env] = False

        # instead of the whole range, we actually are only comparing a window i:i+k for each query/target i - for each, select the appropriate k
        # we essentially gather diagonals from this full mask, t of them, k long
        valid_diagonals = [torch.diagonal(valid_modeling_queries, offset=-i) for i in range(t)] # pull the appropriate k per timestep
        valid_mask = torch.stack(valid_diagonals, dim=0).permute(0, 3, 1, 2) # t x n x 1 x k -> t x k x n x 1

        positives_masked_logits = torch.masked_select(positives_logits, valid_mask)
        negatives_masked_logits = torch.masked_select(negatives_logits, valid_mask)
        positive_loss = F.binary_cross_entropy_with_logits(
            positives_masked_logits, torch.ones_like(positives_masked_logits), reduction='none'
        )

        subsampled_positive = self.masked_sample_and_scale(positive_loss)
        negative_loss = F.binary_cross_entropy_with_logits(
            negatives_masked_logits, torch.zeros_like(negatives_masked_logits), reduction='none'
        )
        subsampled_negative = self.masked_sample_and_scale(negative_loss)

        return subsampled_positive + subsampled_negative

@baseline_registry.register_aux_task(name="CPCA_Single")
class CPCA_Single(RolloutAuxTask):
    """ Action-conditional CPC
        From: https://arxiv.org/abs/1811.06407
    """
    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        num_actions = len(task_cfg.POSSIBLE_ACTIONS)
        self.classifier = nn.Sequential(
            nn.Linear(self.aux_hidden_size + cfg.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.action_embedder = nn.Embedding(num_actions + 1, ACTION_EMBEDDING_DIM)
        self.query_gru = nn.GRU(ACTION_EMBEDDING_DIM, self.aux_hidden_size)
        self.k = self.aux_cfg.num_steps

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
        vision = sensor_embeddings["all"]
        # we pass in h_t - 1, a_t - 1 to predict x_t
        k = self.k
        rnn_tk = belief_features[:-k].view(-1, self.aux_hidden_size)
        action_embedding = self.action_embedder(actions) # 128 x 1 x 4, t x n x 4
        # I'm forwarding actions for each timestep, but I shouldn't include the one from the last timestep since nothing is past it
        action_seq = action_embedding[:-1].unfold(0, k, 1).permute(3, 0, 1, 2)\
            .view(k, -1, action_embedding.size(-1)) # k x (t-k+1 -1) x n x -1
        _, query_all = self.query_gru(action_seq, rnn_tk.unsqueeze(0)) # output 1 x (t-k x n) x -1
        query = query_all.squeeze(0).view(t-k, n, self.aux_hidden_size)
        # first k timesteps have no query
        query_padded = torch.cat((torch.zeros(k, n, self.aux_hidden_size, device=self.device), query), dim=0)
        # query is for timesteps 0 -> t - 1 (exactly matching rollout)
        # Targets
        positives = vision
        negative_inds = torch.randperm(t * n, device=self.device)
        negatives = torch.gather(
            positives.view(t * n, -1),
            dim=0,
            index=negative_inds.view(t * n, 1).expand(-1, positives.size(-1)),
        ).view(t, n, -1)
        positives = self.classifier(torch.cat([positives, query_padded], -1))
        negatives = self.classifier(torch.cat([negatives, query_padded], -1))

        valid_modeling_queries = torch.ones(
                t, n, 1, device=self.device, dtype=torch.bool
        )
        valid_modeling_queries[0:k] = False
        for env in range(n):
            has_zeros_batch = env_zeros[env]
            for z in has_zeros_batch:
                valid_modeling_queries[z : z + k, env] = False

        positives = torch.masked_select(positives, valid_modeling_queries)
        negatives = torch.masked_select(negatives, valid_modeling_queries)
        positive_loss = F.binary_cross_entropy_with_logits(
            positives, torch.ones_like(positives), reduction='none'
        )
        subsampled_positive = self.masked_sample_and_scale(positive_loss)
        negative_loss = F.binary_cross_entropy_with_logits(
            negatives, torch.zeros_like(negatives), reduction='none'
        )
        subsampled_negative = self.masked_sample_and_scale(negative_loss)
        return subsampled_positive + subsampled_negative

# ======================
# * New tasks begin
# ======================
@baseline_registry.register_aux_task(name="PBL")
class PBL(RolloutAuxTask):
    """ Prediction of Bootstrapped Latents: https://arxiv.org/pdf/2004.14646.pdf
        Follows quite a similar structure to CPCA

        PBL uses MSE loss to enforce similarity between rollout @ t+k with observation at t+k, and observation should reflect state at t+k.
    """

    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        num_actions = len(task_cfg.POSSIBLE_ACTIONS)
        # Frame - cfg.hidden_size
        # belief - self.aux_hidden_size
        self.forward_classifier = nn.Sequential(
            nn.Linear(self.aux_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, cfg.hidden_size)
        ) # query and perception
        self.reverse_classifier = nn.Sequential(
            nn.Linear(cfg.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, self.aux_hidden_size)
        )
        self.action_embedder = nn.Embedding(num_actions + 1, ACTION_EMBEDDING_DIM)
        self.query_gru = nn.GRU(ACTION_EMBEDDING_DIM, self.aux_hidden_size)
        self.k = self.aux_cfg.num_steps
        self.EPS = 1e-8

    def unroll(self, embedded_seq, n: int):
        # embedded_seq: t x n x -1
        # returns: t x n x -1 x k
        padding = torch.zeros(self.k - 1, n, embedded_seq.size(-1), device=self.device)
        padded = torch.cat((embedded_seq, padding), dim=0) # (t+k-1) x n x -1
        return padded.unfold(dimension=0, size=self.k, step=1)

    # Targets: predict k steps for each starting timestep
    def norm_reg(self, seq): # Norm and regularization as per PBL apdx.
        # seq: t k n h
        norm = torch.norm(seq, dim=-1) # t k n
        norm_loss = 0.02 * (norm.mean() - 1) ** 2
        return seq / (norm.unsqueeze(-1) + self.EPS), norm_loss

    def get_regularized_mse(self, pred, target, valid_mask):
        # With normalization, regularization as dictated in PBL paper
        pred = torch.masked_select(pred, valid_mask.expand_as(pred))
        target = torch.masked_select(target, valid_mask.expand_as(target))
        loss = F.mse_loss(pred, target.detach(), reduction='none')
        subsampled_loss = self.masked_sample_and_scale(loss)
        return subsampled_loss

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
        r"""
            belief_features: t x n x h
        """
        k = self.k

        action_embedding = self.action_embedder(actions)

        action_seq = self.unroll(action_embedding, n).permute(3, 0, 1, 2)\
            .view(k, t*n, action_embedding.size(-1))
        # for each timestep, predict up to k ahead -> (t,k) is query for t + k (+ 1) (since k is 0-indexed) i.e. (0, 0) -> query for 1
        out_all, _ = self.query_gru(action_seq, belief_features.view(t*n, -1).unsqueeze(0))
        query_all = out_all.view(k, t, n, -1).permute(1, 0, 2, 3) # so, t k n h

        obs_seq = self.unroll(sensor_embeddings["all"], n).permute(0, 3, 1, 2) # t k n h
        obs_seq, obs_reg_loss = self.norm_reg(obs_seq)

        belief_seq = self.unroll(belief_features, n).permute(0, 3, 1, 2)
        forward_pred = self.forward_classifier(query_all)
        forward_pred, forward_reg_loss = self.norm_reg(forward_pred)
        forward_target = obs_seq # We need to stack these for k

        reverse_pred = self.reverse_classifier(obs_seq)
        reverse_target = belief_seq

        # Masking (same as CPCA)
        # Note which timesteps [1, t+k+1] could have valid queries, at distance (k) (note offset by 1)
        valid_modeling_queries = torch.ones(
            t + k, k, n, 1, device=self.device, dtype=torch.bool
        ) # (padded) timestep predicted x prediction distance x env
        valid_modeling_queries[t - 1:] = False # >= t is past rollout, and t is index t - 1 here
        for j in range(1, k + 1): # for j-step predictions
            valid_modeling_queries[:j - 1, j - 1] = False # first j frames cannot be valid for all envs (rollout doesn't go that early)
            for env in range(n):
                has_zeros_batch = env_zeros[env]
                # in j-step prediction, timesteps z -> z + j are disallowed as those are the first j timesteps of a new episode
                # z-> z-1 because of modeling_queries being offset by 1
                for z in has_zeros_batch:
                    valid_modeling_queries[z-1: z-1 + j, j - 1, env] = False

        # instead of the whole range, we actually are only comparing a window i:i+k for each query/target i - for each, select the appropriate k
        # we essentially gather diagonals from this full mask, t of them, k long
        valid_diagonals = [torch.diagonal(valid_modeling_queries, offset=-i) for i in range(t)] # pull the appropriate k per timestep
        valid_mask = torch.stack(valid_diagonals, dim=0).permute(0, 3, 1, 2) # t x n x 1 x k -> t x k x n x 1

        forward = self.get_regularized_mse(forward_pred, forward_target, valid_mask)
        reverse = self.get_regularized_mse(reverse_pred, reverse_target, valid_mask)

        return forward + reverse + obs_reg_loss * 0.1 + forward_reg_loss

@baseline_registry.register_aux_task(name="GID")
class GeneralizedInverseDynamics(RolloutAuxTask):
    """ Like Action Prediction, but conditioned on starting and ending frame
        - feed starting frame, ending frame, and starting belief into an initial hidden state
            - h_t, phi_t, phi_t+k -> t -> (0, k-1)
        - we predict the action sequence using a GRU's outputs (as opposed to linear decoding)
            - a_t, a_t+1, ... a_t+k-1 (predicted + labels)
        - provide CEL @ each timestep
        - even though policy is MLP, we're going to encourage easy decoding of short term trajectory
        - closed loop - we provide feedback for each step
    """

    TRAJ_HIDDEN_SIZE = 32

    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        num_actions = len(task_cfg.POSSIBLE_ACTIONS)
        self.k = aux_cfg.num_steps
        assert 0 < self.k <= cfg.num_steps, "GID requires prediction range to be in (0, t]"
        self.initializer = nn.Sequential(
            nn.Linear(2 * cfg.hidden_size + self.aux_hidden_size, 32),
        )
        self.action_embedder = nn.Embedding(num_actions + 1, ACTION_EMBEDDING_DIM)
        self.query_gru = nn.GRU(ACTION_EMBEDDING_DIM, self.TRAJ_HIDDEN_SIZE)
        self.classifier = nn.Sequential(
            nn.Linear(self.TRAJ_HIDDEN_SIZE, num_actions)
        ) # Output logits for actions
        self.cel = nn.CrossEntropyLoss(reduction='none')

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
        vision = sensor_embeddings["all"]

        # Going to do t' = t-k of these
        belief_features = belief_features[:-self.k]
        start_frames = vision[:-self.k] # t' x n x -1
        end_frames = vision[self.k:] # t' x n x -1

        init_input = torch.cat([belief_features, start_frames, end_frames], dim=-1)
        init_hidden = self.initializer(init_input) # (t' x n x -1)
        init_input = init_hidden.view((t-self.k) * n, -1).unsqueeze(0) # 1 x (t'*n) x -1
        action_seq = actions[:-1].unfold(dimension=0, size=self.k, step=1) # t' x n x k (this is the target)
        action_embedding = self.action_embedder(action_seq) # t' x n x k x -1
        action_in = action_embedding[:, :, :-1] # trim final action (not an input, just a target)
        action_in = action_in.permute(2, 0, 1, 3).view(self.k-1, (t-self.k)*n, action_embedding.size(-1))

        out_all, _ = self.query_gru(action_in, init_input)
        query_all = out_all.view(self.k-1, t - self.k, n, -1)
        query_all = torch.cat([init_hidden.unsqueeze(0), query_all], dim=0)
        query_all = query_all.permute(1, 2, 0, 3)
        action_logits = self.classifier(query_all).permute(0,3,1,2) # t' x A x n x k
        pred_loss = self.cel(action_logits, action_seq) # t' x n x k
        # Masking - reject up to k-1 behind a border cross (z-1 is last actual obs)
        valid_modeling_queries = torch.ones(
            t - self.k, n, 1, device=self.device, dtype=torch.bool
        )
        for env in range(n):
            has_zeros_batch = env_zeros[env]
            for z in has_zeros_batch:
                valid_modeling_queries[z-self.k: z, env] = False

        return self.masked_sample_and_scale(pred_loss, mask=valid_modeling_queries)

# * This is referred to as ADP in the paper.
@baseline_registry.register_aux_task(name="ActionDist")
class ActionDist(RolloutAuxTask):
    """ GID, predicting distribution. Easier than counts
        - feed starting belief and ending frame into an initial hidden state
            - h_t, phi_t+k -> action probs
        - KL distribution loss over actions t-> t+k-1
    """

    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        self.num_actions = len(task_cfg.POSSIBLE_ACTIONS)
        self.PREDICTION = 1
        self.RECALL = 0
        self.mode = self.PREDICTION
        self.k = aux_cfg.num_steps # wow, it can be negative!
        if self.k < 0:
            self.mode = self.RECALL
            self.k = -self.k
        self.decoder = nn.Sequential(
            nn.Linear(cfg.hidden_size + self.aux_hidden_size, 32), # MLP probe for action distribution to be taken
            nn.ReLU(),
            nn.Linear(32, self.num_actions)
        )

    # @torch.jit.export
    # Categorical not supported with script
    # https://github.com/pytorch/pytorch/issues/18094
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
        vision = sensor_embeddings["all"]

        # Going to do t' = t-k of these
        if self.mode == self.PREDICTION: # forward
            belief_features = belief_features[:-self.k]
            end_frames = vision[self.k:] # t' x n x -1
        else: # backward
            belief_features = belief_features[self.k:]
            end_frames = vision[:-self.k]

        init_input = torch.cat([belief_features, end_frames], dim=-1).view((t - self.k) * n, -1)
        action_pred = self.decoder(init_input)
        action_pred = Categorical(F.softmax(action_pred, dim=1))
        action_seq = actions[:-1].unfold(dimension=0, size=self.k, step=1) # t' x n x k (this is the target)
        action_seq = action_seq.view((t-self.k) * n, self.k) # (t' x n) over k # needs to be over 4
        # Count the numbers by scattering into one hot and summing
        action_gt = torch.zeros(*action_seq.size(), self.num_actions, device=self.device)
        action_gt.scatter_(dim=-1, index=action_seq.unsqueeze(-1), value=1) # A t indices specified by action seq (actions taken), scatter 1s
        action_gt = action_gt.sum(-2) # (t'*n) x num_actions now, turn to a distribution
        action_gt = Categorical(action_gt.float() / action_gt.sum(-1).unsqueeze(-1))
        pred_loss = kl_divergence(action_gt, action_pred).view(t-self.k, n) # t' x n
        # Masking - reject up to k-1 behind a border cross (z-1 is last actual obs)
        valid_modeling_queries = torch.ones(
            t, n, device=self.device, dtype=torch.bool
        )
        for env in range(n):
            has_zeros_batch = env_zeros[env]
            for z in has_zeros_batch:
                if self.mode == self.PREDICTION:
                    valid_modeling_queries[z-self.k: z, env] = False
                else: # recall? Mask first k frames
                    valid_modeling_queries[z:z+self.k, env] = False
        if self.mode == self.PREDICTION:
            valid_modeling_queries = valid_modeling_queries[:-self.k]
        else:
            valid_modeling_queries = valid_modeling_queries[self.k:]
        return self.masked_sample_and_scale(pred_loss, mask=valid_modeling_queries)

# ======================
# * New tasks end
# ======================

@baseline_registry.register_aux_task(name="CPCA_Weighted")
class CPCA_Weighted(RolloutAuxTask):
    """ To compare with combined aux losses. 5 * k<=1, 4 * k<=2, 3 * k<=4, 2 * k <= 8, 1 * k <= 16 (hardcoded)
        Note - this aux loss is an order of magnitude higher than others (intentionally)
    """
    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        num_actions = len(task_cfg.POSSIBLE_ACTIONS)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_size + self.aux_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ) # query and perception
        self.action_embedder = nn.Embedding(num_actions + 1, ACTION_EMBEDDING_DIM)
        self.query_gru = nn.GRU(ACTION_EMBEDDING_DIM, self.aux_hidden_size)

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
        vision = sensor_embeddings["all"]

        k = 16

        belief_features = belief_features.view(t*n, -1).unsqueeze(0)
        positives = vision
        negative_inds = torch.randperm(t * n, device=self.device)
        negatives = torch.gather(
            positives.view(t * n, -1),
            dim=0,
            index=negative_inds.view(t * n, 1).expand(-1, positives.size(-1)),
        ).view(t, n, -1)
        action_embedding = self.action_embedder(actions) # t n -1
        action_padding = torch.zeros(k - 1, n, action_embedding.size(-1), device=self.device)
        action_padded = torch.cat((action_embedding, action_padding), dim=0) # (t+k-1) x n x -1
        # t x n x -1 x k
        action_seq = action_padded.unfold(dimension=0, size=k, step=1).permute(3, 0, 1, 2)\
            .view(k, t*n, action_embedding.size(-1))

        # for each timestep, predict up to k ahead -> (t,k) is query for t + k (+ 1) (since k is 0-indexed) i.e. (0, 0) -> query for 1
        out_all, _ = self.query_gru(action_seq, belief_features)
        query_all = out_all.view(k, t, n, -1).permute(1, 0, 2, 3)

        # Targets: predict k steps for each starting timestep
        positives_padded = torch.cat((positives[1:], torch.zeros(k, n, positives.size(-1), device=self.device)), dim=0) # (t+k) x n
        positives_expanded = positives_padded.unfold(dimension=0, size=k, step=1).permute(0, 3, 1, 2) # t x k x n x -1
        positives_logits = self.classifier(torch.cat([positives_expanded, query_all], -1))
        negatives_padded = torch.cat((negatives[1:], torch.zeros(k, n, negatives.size(-1), device=self.device)), dim=0) # (t+k) x n x -1
        negatives_expanded = negatives_padded.unfold(dimension=0, size=k, step=1).permute(0, 3, 1, 2) # t x k x n x -1
        negatives_logits = self.classifier(torch.cat([negatives_expanded, query_all], -1))

        # Masking
        # Note which timesteps [1, t+k+1] could have valid queries, at distance (k) (note offset by 1)
        valid_modeling_queries = torch.ones(
            t + k, k, n, 1, device=self.device, dtype=torch.bool # not uint so we can mask with this
        ) # (padded) timestep predicted x prediction distance x env
        valid_modeling_queries[t - 1:] = False # >= t is past rollout, and t is index t - 1 here
        for j in range(1, k + 1): # for j-step predictions
            valid_modeling_queries[:j - 1, j - 1] = False # first j frames cannot be valid for all envs (rollout doesn't go that early)
            for env in range(n):
                has_zeros_batch = env_zeros[env]
                # in j-step prediction, timesteps z -> z + j are disallowed as those are the first j timesteps of a new episode
                # z-> z-1 because of modeling_queries being offset by 1
                for z in has_zeros_batch:
                    valid_modeling_queries[z-1: z-1 + j, j - 1, env] = False

        # instead of the whole range, we actually are only comparing a window i:i+k for each query/target i - for each, select the appropriate k
        # we essentially gather diagonals from this full mask, t of them, k long
        valid_diagonals = [torch.diagonal(valid_modeling_queries, offset=-i) for i in range(t)] # pull the appropriate k per timestep
        valid_mask = torch.stack(valid_diagonals, dim=0).permute(0, 3, 1, 2) # t x n x 1 x k -> t x k x n x 1

        weight_mask = torch.tensor([5, 4, 3, 3, 2, 2, 2, 2,
                                    1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32,
                                    device=self.device) # this should be multiplied on the loss
        # mask over the losses, not the logits
        positive_loss = F.binary_cross_entropy_with_logits(
            positives_logits, torch.ones_like(positives_logits), reduction='none'
        ) # t k n 1 still
        positive_loss = positive_loss.permute(0, 2, 3, 1) * weight_mask # now t n 1 k
        subsampled_positive = self.masked_sample_and_scale(positive_loss.permute(0, 3, 1, 2), mask=valid_mask)
        negative_loss = F.binary_cross_entropy_with_logits(
            negatives_logits, torch.zeros_like(negatives_logits), reduction='none'
        )
        negative_loss = negative_loss.permute(0, 2, 3, 1) * weight_mask
        subsampled_negative = self.masked_sample_and_scale(negative_loss.permute(0, 3, 1, 2), mask=valid_mask)
        return subsampled_positive + subsampled_negative

# Clones of the CPC|A task so that we can allow different task parameters under yacs
# Used to run k=1, 2, 4, 8, 16, in the same exp
@baseline_registry.register_aux_task(name="CPCA_A")
class CPCA_A(CPCA):
    pass

@baseline_registry.register_aux_task(name="CPCA_B")
class CPCA_B(CPCA):
    pass

@baseline_registry.register_aux_task(name="CPCA_C")
class CPCA_C(CPCA):
    pass

@baseline_registry.register_aux_task(name="CPCA_D")
class CPCA_D(CPCA):
    pass

@baseline_registry.register_aux_task(name="CPCA_Single_A")
class CPCA_Single_A(CPCA_Single):
    pass

@baseline_registry.register_aux_task(name="CPCA_Single_B")
class CPCA_Single_B(CPCA_Single):
    pass

@baseline_registry.register_aux_task(name="CPCA_Single_C")
class CPCA_Single_C(CPCA_Single):
    pass

@baseline_registry.register_aux_task(name="CPCA_Single_D")
class CPCA_Single_D(CPCA_Single):
    pass

@baseline_registry.register_aux_task(name="ActionDist_A")
class ActionDist_A(ActionDist):
    pass

@baseline_registry.register_aux_task(name="Dummy")
class DummyTask(RolloutAuxTask):
    r""" 0-loss auxiliary task"""
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
        return torch.tensor(0.0, device=self.device)

