#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

import torch

class RolloutStorage:
    r"""Class for storing rollout information for RL trainers.

    """

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        num_recurrent_memories=1,
        num_policy_heads=1,
        metrics=[], # list of metric scalars we want to track
    ):
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )
        if 'semantic' not in self.observations:
            self.observations['semantic'] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces['depth'].shape[:2] # No channel dimension
            )

        # * Note: these modules are hidden in the rollout, i.e. if an architecture uses one module the rollout API does not expect the module dimension
        # * This is to be compatible with baseline architectures
        self.num_modules = num_recurrent_memories
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1,
            num_recurrent_layers,
            num_envs,
            self.num_modules,
            recurrent_hidden_state_size,
        )

        # Policy dim is stored in dim 2 to not interfere with flattening + recurrent generator code.
        # As a tradeoff, we fiddle with masking a bit.
        self.rewards = torch.zeros(num_steps, num_envs, num_policy_heads, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, num_policy_heads, 1)

        self.value_preds = torch.zeros(num_steps + 1, num_envs, num_policy_heads, 1)
        self.action_log_probs = torch.zeros(num_steps, num_envs, num_policy_heads, 1)

        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        self.prev_actions = torch.zeros(num_steps + 1, num_envs, action_shape)
        if action_space.__class__.__name__ == "ActionSpace":
            self.actions = self.actions.long()
            self.prev_actions = self.prev_actions.long()

        self.metrics = {}
        for metric in metrics:
            self.metrics[metric] = torch.zeros(
                num_steps,
                num_envs, # all scalars
            )

        self.masks = torch.zeros(num_steps + 1, num_envs, 1, dtype=torch.bool)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        for m in self.metrics:
            self.metrics[m] = self.metrics[m].to(device)
        self.masks = self.masks.to(device)

    def to_fp16(self):
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(dtype=torch.float16)
        for sensor in self.observations:
            reading = self.observations[sensor]
            if reading.dtype == torch.float32:
                self.observations[sensor] = reading.to(dtype=torch.float16)

    def insert(
        self,
        observations,
        recurrent_hidden_states,
        actions,
        action_log_probs, # b x k x 1
        value_preds, # b x k x 1
        rewards, # k x b x 1
        masks,
        metrics,
    ):
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )

        if self.num_modules == 1:
            recurrent_hidden_states = recurrent_hidden_states.unsqueeze(-2)
        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )
        self.actions[self.step].copy_(actions)
        self.prev_actions[self.step + 1].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards.permute(1, 0, 2))
        for m in metrics:
            self.metrics[m][self.step].copy_(metrics[m])
        self.masks[self.step + 1].copy_(masks)

        self.step = self.step + 1

    def get_recurrent_states(self):
        if self.num_modules == 1:
            return self.recurrent_hidden_states.squeeze(-2)
        return self.recurrent_hidden_states

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(
                self.observations[sensor][self.step]
            )

        self.recurrent_hidden_states[0].copy_(
            self.recurrent_hidden_states[self.step]
        )
        self.masks[0].copy_(self.masks[self.step])
        self.prev_actions[0].copy_(self.prev_actions[self.step])
        self.step = 0

    def compute_returns(self, next_value,
        use_gae, gamma, tau,
        behavioral_index=0, importance_weight=False, weight_clip=[0.01, 1.0]
    ):
        if len(next_value.size()) == 2:
            next_value = next_value.unsqueeze(-1)
        if use_gae:
            self.value_preds[self.step] = next_value # b x k x 1
            gae = 0
            if importance_weight: # Note we let rho = 1, to match target policy.
                # gae structure looks remarkably similar to vtrace.
                # Follow it, but sub out importance samples for target policy.
                iw = torch.exp(
                    self.action_log_probs
                    - self.action_log_probs[:, :, behavioral_index].unsqueeze(2)
                )
                iw_clipped = torch.clamp(iw, weight_clip[0], weight_clip[1])
                tau = tau * iw_clipped
            for step in reversed(range(self.step)):
                mask = self.masks[step + 1].unsqueeze(1) # b x 1 -> b x 1 x 1
                delta = (
                    self.rewards[step] # b x k x 1
                    + gamma * self.value_preds[step + 1] * mask
                    - self.value_preds[step]
                )
                tau_step = tau[step] if importance_weight else tau
                gae = delta + gamma * tau_step * mask * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[self.step] = next_value
            for step in reversed(range(self.step)):
                mask = self.masks[step + 1].unsqueeze(1) # b x 1 -> b x 1 x 1
                self.returns[step] = (
                    self.returns[step + 1] * gamma * mask
                    + self.rewards[step]
                )

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            metrics_batch = defaultdict(list)

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset] # the env

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][: self.step, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0, :, ind] # add the first hidden state
                )

                actions_batch.append(self.actions[: self.step, ind])
                prev_actions_batch.append(self.prev_actions[: self.step, ind])
                value_preds_batch.append(self.value_preds[: self.step, ind])
                return_batch.append(self.returns[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[: self.step, ind]
                )
                for m in self.metrics:
                    metrics_batch[m].append(self.metrics[m][: self.step, ind])

                adv_targ.append(advantages[: self.step, ind])

            T, N = self.step, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )
            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1) # T x N x 1
            value_preds_batch = torch.stack(value_preds_batch, 1) # T x N x num_policy x 1
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )
            for m in self.metrics:
                metrics_batch[m] = torch.stack(metrics_batch[m], 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is a (num_recurrent_layers, N, k, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1)
            if self.num_modules == 1:
                recurrent_hidden_states_batch = recurrent_hidden_states_batch.squeeze(-2)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )

            actions_batch = self._flatten_helper(T, N, actions_batch)
            prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = self._flatten_helper(
                T, N, old_action_log_probs_batch
            )
            for m in self.metrics:
                metrics_batch[m] = self._flatten_helper(T, N, metrics_batch[m])

            adv_targ = self._flatten_helper(T, N, adv_targ)

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                metrics_batch,
                adv_targ,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])