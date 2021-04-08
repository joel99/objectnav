#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn

from habitat_baselines.common.auxiliary_tasks.aux_utils import (
    ACTION_EMBEDDING_DIM
)

"""
This curiosity work an experimental intrinsic curiosity reward;
However we found it made little diff.
"""

class ForwardCuriosity(nn.Module):
    r"""
        Curiosity as described in Pathak et al (2018) https://pathak22.github.io/noreward-rl/
        We can't afford a separate CNN to encode the state, nor does it seem desirable, so we share with the main agent.
        Input: env representation as input i.e. the sensor state.
        We'll use an inverse step directly on the obs embedding to transform to a repr more informative about agent action consequences
        and predict in that space.

        Addition: We add belief
        - motivation: agent fidgets in high-entropy areas in train + val, but this is never productive
        - by adding memory to the decoding, the agent should be much less curious about areas it has seen in current trajectory
        - we don't use in inverse decoding as it would be trivial for belief to project action to take
            - which would be useless for modeling environment dynamics in inverse space
    """
    def __init__(self, cfg, task_cfg, embedding_size):
        super().__init__()
        num_actions = len(task_cfg.POSSIBLE_ACTIONS)

        hidden_size = cfg.hidden_size
        self.use_inverse_space = cfg.CURIOSITY.USE_INVERSE_SPACE
        self.use_beliefs = cfg.CURIOSITY.USE_BELIEF
        self.inverse_beta = cfg.CURIOSITY.INVERSE_BETA
        self.loss_scale = cfg.CURIOSITY.LOSS_SCALE

        fwd_in_features = ACTION_EMBEDDING_DIM
        if self.use_beliefs:
            fwd_in_features += hidden_size

        if self.use_inverse_space:
            self.inverse = nn.Linear(embedding_size, hidden_size)
            self.action_pred = nn.Linear(2 * hidden_size, num_actions)
            self.action_classifier = nn.CrossEntropyLoss(reduction='none')
            self.fwd_predictor = nn.Sequential(
                nn.Linear(fwd_in_features + hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )
        else:
            self.fwd_predictor = nn.Sequential(
                nn.Linear(fwd_in_features + embedding_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, embedding_size)
            )

        self.action_embedder = nn.Embedding(num_actions + 1, ACTION_EMBEDDING_DIM)

    def forward(self, s, s_next, actions, beliefs=None, masks=None):
        r"""
            s: b x h
            s_next: b x h
            action: b x 1
            beliefs: some module's belief or the aggregate belief. b x h
            masks: b x 1
            Returns
                forward prediction error
                total loss
        """
        # Inverse step
        inverse_loss = torch.tensor(0, dtype=torch.float, device=s.device)
        if self.use_inverse_space:
            inverse_features = torch.stack([s, s_next], dim=0)
            inverse_features = self.inverse(inverse_features) # 2 x b x h
            s = inverse_features[0]
            s_next = inverse_features[1]
            inverse_action_in = torch.cat([s, s_next], dim=-1)
            inverse_preds = self.action_pred(inverse_action_in)
            inverse_loss = self.action_classifier(inverse_preds, actions.squeeze(-1))
        # Normalize state features so curiosity is bounded
        s = s / s.norm(dim=-1).unsqueeze(-1)
        s_next = s_next / s_next.norm(dim=-1).unsqueeze(-1)
        action_embedding = self.action_embedder(actions.squeeze(-1)) # b x 4
        decoder_in = [action_embedding, s]
        if self.use_beliefs:
            decoder_in.append(beliefs)
        decoder_in = torch.cat(decoder_in, dim=-1) # b x -1
        preds = self.fwd_predictor(decoder_in)
        pred_error = 0.5 * (s_next - preds).pow(2).sum(1)

        # loss is b x -1
        # if mask is 0 at that slot, there's been an episode change, so zero the loss
        if masks is not None:
            pred_error = pred_error * masks
            inverse_loss = inverse_loss * masks

        inv_scale = (1 - self.inverse_beta) if self.use_inverse_space else 1
        pred_loss = pred_error * self.loss_scale * inv_scale
        return (
            pred_error,
            self.inverse_beta * inverse_loss * self.loss_scale,
            pred_loss
        )