#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Tuple, List
import contextlib

import torch
from torch import nn as nn
from torch import optim as optim

from habitat import logger
from habitat_baselines.common.auxiliary_tasks.auxiliary_tasks import ActionDist, ActionDist_A
from habitat_baselines.rl.ddppo.algo.ddp_utils import rank0_only
EPS_PPO = 1e-5

class FP16OptimParamManager:
    def __init__(self, optimizer: optim.Optimizer):
        self.optimizer = optimizer

        self._fp16_params = []
        self._fp32_params = []

        for pg in optimizer.param_groups:
            new_fp32_params = []
            self._fp16_params.append(pg["params"])
            for param in pg["params"]:
                fp32_param = (
                    param.data.to(dtype=torch.float32)
                    if param.dtype == torch.float16
                    else param
                )

                new_fp32_params.append(fp32_param)

            pg["params"] = new_fp32_params
            self._fp32_params.append(new_fp32_params)

    def _apply_fn(self, function):
        for fp32_pg, fp16_pg in zip(self._fp32_params, self._fp16_params):
            for fp32_p, fp16_p in zip(fp32_pg, fp16_pg):
                function(fp32_p, fp16_p)

    def sync_grads(self):
        def _sync_grad_fn(fp32_p, fp16_p):
            if fp16_p.grad is not None:
                fp32_p.grad = (
                    fp16_p.grad.data.to(dtype=torch.float32)
                    if fp16_p.grad.dtype == torch.float16
                    else fp16_p.grad.data
                )

        self._apply_fn(_sync_grad_fn)

    def sync_params(self):
        self._apply_fn(lambda fp32_p, fp16_p: fp16_p.data.copy_(fp32_p.data))

    def set_fp32_params_from_optim(self):
        self._fp32_params = [
            pg["params"] for pg in self.optimizer.param_groups
        ]
        self.sync_params()

    def clear_grads(self):
        def _set_none(fp32_p, fp16_p):
            fp32_p.grad = None
            fp16_p.grad = None

        self._apply_fn(_set_none)


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        aux_loss_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,
        aux_tasks=[],
        aux_cfg=None,
        aux_encoders={},
        aux_map=None,
        curiosity_cfg=None,
        curiosity_module=None,
        importance_weight=False,
        fp16_autocast: bool = False,
        fp16_mixed: bool = False,
    ):

        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.importance_weight = importance_weight

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.aux_loss_coef = aux_loss_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage
        self.aux_tasks = nn.ModuleList(aux_tasks)
        self.aux_encoders = nn.ModuleDict(aux_encoders) # it'd be nice to script these
        self.aux_cfg = aux_cfg
        self.aux_map = aux_map

        self.curiosity_cfg = curiosity_cfg
        if self.should_use_curiosity():
            assert curiosity_module is not None
        self.curiosity_module = curiosity_module

        self.optimizer = optim.AdamW(
            list(filter(lambda p: p.requires_grad, self.get_parameters())),
            lr=lr,
            eps=eps,
            weight_decay=1e-4 if fp16_autocast or fp16_mixed else 0.0,
        )

        self.grad_scaler = (
            torch.cuda.amp.GradScaler()
            if fp16_autocast or fp16_mixed
            else None
        )
        self._fp16_autocast = fp16_autocast
        self._fp16_mixed = fp16_mixed

        self._consecutive_steps_with_scale_reduce = 0
        self._prev_grad_scale = (
            self.grad_scaler.get_scale()
            if self.grad_scaler is not None
            else None
        )

        self.fp16_optim_params = (
            FP16OptimParamManager(self.optimizer) if fp16_mixed else None
        )

    def script(self):
        # This is very haphazardly implemented
        self.aux_tasks = nn.ModuleList(
            (task if isinstance(task, ActionDist) or isinstance(task, ActionDist_A) else torch.jit.script(task))\
                for task in self.aux_tasks
        )
        # torch.jit.script(self.aux_tasks) doesn't work directly
        # self.actor_critic.visual_encoders.encoders = torch.jit.script(self.actor_critic.visual_encoders.encoders)

    def get_parameters(self):
        params = list(self.actor_critic.parameters())
        for task in self.aux_tasks:
            params += list(task.parameters())
        for encoder in self.aux_encoders.values():
            params += list(encoder.parameters())
        if self.should_use_curiosity():
            params += list(self.curiosity_module.parameters())
        return params

    def optim_state_dict(self):
        return dict(
            optimizer=self.optimizer.state_dict(),
            grad_scaler=self.grad_scaler.state_dict()
            if self.grad_scaler is not None
            else None,
        )

    def load_optim_state(self, optim_state, is_warm_start=False):
        optim_only_dict = optim_state
        if "optimizer" in optim_state:
            optim_only_dict = optim_state["optimizer"] # See optim_state_dict() -- otherwise, this state is the whole deal
        if is_warm_start:
            # * This code is for loading weights + optimizer state into a slightly modified arch.
            # At this point in time, optimizer doesn't appear to have any state
            # So we only remove extra params and re-insert them once checkpoint is loaded
            # We currently assume that new params are at end of belief policy, but that's p arbitrary
            # ! HARDCODING reflects order in ppo.get_parameters()
            # ! HARDCODING also reflects only 1 param group
            # ! HARCODING slice indices retrieved from self.actor_critic.parameters() order
            new_module_key = "message_encoder"
            new_module_mask = list(map(lambda pair: new_module_key in pair[0], self.actor_critic.named_parameters()))
            if any(new_module_mask):
                # Surgical removal. Assumes contiguous new params
                new_module_first = new_module_mask.index(True)
                new_module_last = len(new_module_mask) - 1 - new_module_mask[::-1].index(True)
                all_params = self.optimizer.param_groups[0]['params']
                new_params = all_params[new_module_first:new_module_last+1]
                old_params = all_params[:new_module_first] + all_params[new_module_last+1:]
                self.optimizer.param_groups[0]['params'] = old_params
            self.optimizer.load_state_dict(optim_only_dict)
            # Re-insert
            if any(new_module_mask):
                updated_params = self.optimizer.param_groups[0]['params']
                self.optimizer.param_groups[0]['params'] = updated_params[:new_module_first] + new_params + updated_params[new_module_first:]
        else:
            self.optimizer.load_state_dict(optim_only_dict)

        if self.grad_scaler is not None:
            self.grad_scaler.load_state_dict(optim_state["grad_scaler"])

        if self.fp16_optim_params is not None:
            self.fp16_optim_params.set_fp32_params_from_optim()


    def should_use_curiosity(self):
        return self.curiosity_cfg is not None and self.curiosity_cfg.USE_CURIOSITY

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        # ^ t x n x k x 1 or t x n x 1
        if not self.use_normalized_advantage:
            return advantages
        flat_adv = advantages.flatten(end_dim=1) # b x k x 1, or b x 1
        mean = flat_adv.mean(dim=0) # k x 1, or 1
        std = flat_adv.std(dim=0)
        return (advantages - mean) / (std + EPS_PPO)


    def get_curiosity_loss(self, *args, **kwargs):
        return self.curiosity_module(*args, **kwargs)[1:]

    def get_curiosity_error(self, *args, **kwargs):
        return self.curiosity_module(*args, **kwargs)[0]

    def _shape_aux_inputs(
        self,
        actions,
        masks,
        metrics: Dict[str, torch.Tensor],
        final_rnn_state,
        vision_embeddings: Dict[str, torch.Tensor],
    ):
        r"""
            Shape rollout into an unflattened format for auxiliary task calculation.
            Returns:
                actions - t x n
                all_embeddings - Dict[str, torch.Tensor t x n x h]
                n - num envs
                t - timesteps
                env_zeros - episode transition timesteps (nested list, env, then steps)
                metrics - t x n
        """
        n = final_rnn_state.size(1)
        masks = masks.view(-1, n)
        env_zeros = [] # Episode crossings per env, lots of tasks use this
        for env in range(n):
            env_zeros.append(
                (masks[:, env] == 0.0).nonzero(as_tuple=False).squeeze(-1).cpu().unbind(0)
            )
        t = masks.size(0)
        actions = actions.view(t, n)
        for m in metrics:
            metrics[m] = metrics[m].view(t, n)
        vision_embeddings = {sensor: rep.view(t, n, -1) for sensor, rep in vision_embeddings.items()}

        return actions, metrics, n, t, env_zeros, vision_embeddings

    def get_aux_losses(
        self,
        sample, observations,
        vision_embeddings, other_embeddings,
        final_rnn_state, rnn_features, individual_rnn_features
    ):
        r"""
            Calculate auxiliary losses. Prepare observations, and pipes correctly.
            args:
                sample: rollout sample

                There's a preprocess, and then we just directly pass to encoders.
            returns:
                individual and total aux loss
        """
        if not self.aux_tasks:
            return [], 0
        # Going to take a gamble that our aux tasks are learned relatively stably, so we can use low precision
        with torch.cuda.amp.autocast(enabled=self._fp16_autocast):
            # aux embeddings are visual embeddings that are only used for auxiliary tasks
            aux_embeddings = { k: encoder(sample[0]) for k, encoder in self.aux_encoders.items() }
            vision_embeddings.update(aux_embeddings)

            actions, metrics, n, t, env_zeros, vision_embeddings = self._shape_aux_inputs(
                sample[2], sample[6], sample[8], final_rnn_state, vision_embeddings
            )

            final_belief_state = final_rnn_state[-1] # * Only use top layer belief state
            output_drop = self.actor_critic.output_drop

            if self.actor_critic.IS_MULTIPLE_BELIEF:
                losses = [None] * len(self.aux_tasks) # In case we want to do something parallel
                for i, task in enumerate(self.aux_tasks):
                    get_belief_index = lambda j: self.aux_map[j] if self.aux_map else j
                    losses[i] = \
                        task.get_loss(
                            observations,
                            actions,
                            vision_embeddings,
                            final_belief_state[:, get_belief_index(i)].contiguous(),
                            output_drop(individual_rnn_features[:, get_belief_index(i)]).contiguous().view(t,n,-1),
                            metrics,
                            n, t, env_zeros
                        )
            else: # single belief
                belief_features = rnn_features.view(t, n, -1)
                belief_features = output_drop(belief_features)
                losses = [task.get_loss(
                    observations,
                    actions,
                    vision_embeddings,
                    final_belief_state,
                    belief_features,
                    metrics,
                    n, t, env_zeros
                ) for task in self.aux_tasks]

            aux_losses = torch.stack(losses)
        return aux_losses, torch.sum(aux_losses, dim=0)

    def update(self, rollouts, gamma, behavioral_index=0):
        r"""
            returns:
            - value_loss_epoch (tensor of either one loss or num_head losses)
            - action_loss_epoch (tensor of either one loss or num_head losses)
        """
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = torch.zeros(advantages.size()[2:3]) # Either k or 1
        action_loss_epoch = torch.zeros(advantages.size()[2:3])
        dist_entropy_epoch = torch.zeros(advantages.size()[2:3])

        aux_losses_epoch = [0] * len(self.aux_tasks)
        aux_entropy_epoch = 0
        aux_weights_epoch = [0] * len(self.aux_tasks)
        inv_curiosity_epoch = 0
        fwd_curiosity_epoch = 0
        for e in range(self.ppo_epoch):
            # This data generator steps through the rollout (gathering n=batch_size processes rollouts)
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch,
            )

            for sample in data_generator:
                with torch.cuda.amp.autocast(enabled=self._fp16_autocast):
                    (
                        obs_batch,
                        recurrent_hidden_states_batch,
                        actions_batch,
                        prev_actions_batch,
                        value_preds_batch,
                        return_batch,
                        masks_batch,
                        old_action_log_probs_batch,
                        _,
                        adv_targ,
                    ) = sample

                    (
                        values,
                        action_log_probs,
                        dist_entropy,
                        final_rnn_state, # Used to encourage trajectory memory (it's the same as final rnn feature due to GRU)
                        rnn_features,
                        individual_rnn_features,
                        aux_dist_entropy,
                        aux_weights,
                        observations,
                        vision_embeddings,
                        other_embeddings,
                    ) = self.actor_critic.evaluate_actions(
                        obs_batch,
                        recurrent_hidden_states_batch,
                        prev_actions_batch,
                        masks_batch,
                        actions_batch,
                    )

                    if self._fp16_mixed: # Cast back to f32 because we're spooked by half-precision opt
                        # FYI we can probably push this through the aux
                        values = values.float()
                        action_log_probs = action_log_probs.float()
                        dist_entropy = dist_entropy.float()
                        final_rnn_state = final_rnn_state.float()
                        rnn_features = rnn_features.float()
                        individual_rnn_features = individual_rnn_features.float()
                        aux_dist_entropy = aux_dist_entropy.float()
                        observations = { k: v.float() for k, v in observations.items() }
                        vision_embeddings = { k: v.float() for k, v in vision_embeddings.items() }
                        other_embeddings = [ v.float() for v in other_embeddings ]

                    # make single policy outputs look like multi-policy
                    # (we preserve single-policy API for policy so agent can be exported more easily)
                    if len(action_log_probs.size()) == 2:
                        # [txn, 1] -> [txn, k, 1]
                        action_log_probs = action_log_probs.unsqueeze(1)
                        values = values.unsqueeze(1)
                        dist_entropy = dist_entropy.unsqueeze(0) # scalar -> 1
                    else: # We already have multi-policy, consider weighting
                        if self.importance_weight:
                            # Update returns to reflect true actor
                            # If we want to weight returns on every ppo update (as more theoretically motivated), we should be doing it here
                            # However, it's inefficient, and we still need to deal with the fact that there's only a subset of envs that's abstracted away that's being used here...
                            # Instead we just update once per PPO cycle in the return calculation
                            old_action_log_probs_batch[:, 1-behavioral_index] = old_action_log_probs_batch[:, behavioral_index]
                    dist_entropy_loss = dist_entropy.mean()

                    # action_log_probs: [txn (, k), 1]

                    ratio = torch.exp(
                        action_log_probs - old_action_log_probs_batch
                    )
                    surr1 = ratio * adv_targ
                    surr2 = (
                        torch.clamp(
                            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                        )
                        * adv_targ
                    )

                    def loss_per_head_and_total(loss):
                        # returns losses [k] and mean loss (scalar)
                        losses = loss.permute(0, 2, 1).flatten(end_dim=1).mean(dim=0) # k
                        # ! discard the first head once we are on the second one (still report both)
                        if behavioral_index == 1:
                            return losses, losses[..., 1]
                        return losses, losses.mean()

                    # We average across multi-policy heads, effective LR for heads is halved
                    action_loss = -torch.min(surr1, surr2)
                    action_losses, action_loss = loss_per_head_and_total(action_loss) # 128 x k x 1

                    # Value loss is MSE with actual(TM) value/rewards
                    if self.use_clipped_value_loss:
                        value_pred_clipped = value_preds_batch + (
                            values - value_preds_batch
                        ).clamp(-self.clip_param, self.clip_param)
                        value_losses = (values - return_batch).pow(2)
                        value_losses_clipped = (
                            value_pred_clipped - return_batch
                        ).pow(2)
                        value_loss = (
                            0.5
                            * torch.max(value_losses, value_losses_clipped)
                        )
                    else:
                        value_loss = 0.5 * (return_batch - values).pow(2)
                    value_losses, value_loss = loss_per_head_and_total(value_loss)

                    aux_losses, total_aux_loss = self.get_aux_losses(
                        sample, observations,
                        vision_embeddings, other_embeddings,
                        final_rnn_state, rnn_features, individual_rnn_features,
                    )

                    total_loss = (
                        value_loss * self.value_loss_coef
                        + action_loss
                        + total_aux_loss * self.aux_loss_coef
                        - dist_entropy_loss * self.entropy_coef
                    )
                    if aux_dist_entropy is not None:
                        total_loss -= aux_dist_entropy * self.aux_cfg.entropy_coef

                    if self.should_use_curiosity():
                        n = final_rnn_state.size(1)
                        curiosity_obs = vision_embeddings[self.curiosity_cfg.VISION_KEY] # TODO we can re-support other sensors if we'd like.
                        input_features = curiosity_obs.detach() if self.curiosity_cfg.BLOCK_ENCODER_GRADIENTS else curiosity_obs
                        actions_cur = actions_batch[:-n].detach() if self.curiosity_cfg.BLOCK_ENCODER_GRADIENTS else actions_batch
                        curiosity_beliefs = None
                        if self.curiosity_cfg.USE_BELIEF:
                            curiosity_beliefs = rnn_features[:-n].detach() if self.curiosity_cfg.BLOCK_ENCODER_GRADIENTS else rnn_features[:-n]
                        inv_curiosity_loss, fwd_curiosity_loss = self.get_curiosity_loss(
                            input_features[:-n], # observations at 0:T-1
                            input_features[n:], # observations at 1:T
                            actions_cur, # actions at 0:T-1
                            beliefs=curiosity_beliefs, # beliefs at 0:T-1
                            masks=masks_batch[n:] # masks at 0:T
                        )
                        inv_curiosity_loss = inv_curiosity_loss.mean()
                        fwd_curiosity_loss = fwd_curiosity_loss.mean()
                        total_loss += inv_curiosity_loss
                        total_loss += fwd_curiosity_loss

                    self.before_backward(total_loss)
                    if self.grad_scaler is not None:
                        total_loss = self.grad_scaler.scale(total_loss)
                    total_loss.backward()
                    self.after_backward(total_loss)

                    self.before_step()
                    if self.grad_scaler is None:
                        self.optimizer.step()
                    else:
                        self.grad_scaler.step(self.optimizer)
                    self.after_step()
                    value_loss_epoch += value_losses.cpu()
                    action_loss_epoch += action_losses.cpu()
                    dist_entropy_epoch += dist_entropy.cpu()
                    if aux_dist_entropy is not None:
                        aux_entropy_epoch += aux_dist_entropy.item()
                    for i, aux_loss in enumerate(aux_losses):
                        aux_losses_epoch[i] += aux_loss.item()
                    if aux_weights is not None:
                        for i, aux_weight in enumerate(aux_weights):
                            aux_weights_epoch[i] += aux_weight.item()
                    if self.should_use_curiosity():
                        inv_curiosity_epoch += inv_curiosity_loss.item()
                        fwd_curiosity_epoch += fwd_curiosity_loss.item()

        self.actor_critic.after_update()
        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        for i, aux_loss in enumerate(aux_losses):
            aux_losses_epoch[i] /= num_updates
        if aux_weights is not None:
            for i, aux_weight in enumerate(aux_weights):
                aux_weights_epoch[i] /= num_updates
        else:
            aux_weights_epoch = None
        if self.should_use_curiosity():
            inv_curiosity_epoch /= num_updates
            fwd_curiosity_epoch /= num_updates

        return (
            value_loss_epoch,
            action_loss_epoch,
            dist_entropy_epoch,
            aux_losses_epoch,
            aux_entropy_epoch,
            aux_weights_epoch,
            inv_curiosity_epoch,
            fwd_curiosity_epoch,
        )

    def before_backward(self, loss):
        pass

    def after_backward(self, loss: torch.tensor) -> None:
        if self.fp16_optim_params is not None:
            self.fp16_optim_params.sync_grads()
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)

    def before_step(self):
        nn.utils.clip_grad_norm_(
            self.get_parameters(), self.max_grad_norm
        )

    def after_step(self) -> None:
        if self.fp16_optim_params is not None:
            self.fp16_optim_params.sync_params()
            self.fp16_optim_params.clear_grads()
        else:
            self.optimizer.zero_grad()

        if self.grad_scaler is not None:
            self.grad_scaler.update()
            new_scale = self.grad_scaler.get_scale()
            if new_scale < self._prev_grad_scale:
                self._consecutive_steps_with_scale_reduce += 1
            else:
                self._consecutive_steps_with_scale_reduce = 0

            self._prev_grad_scale = new_scale
            if self._consecutive_steps_with_scale_reduce > 2 and rank0_only():
                logger.warn(
                    "Greater than 2 steps with scale reduction."
                    "  This typically indicates that fp16 training is unstable."
                    "  Consider switching from mixed to autocast or autocast to off"
                )
            if new_scale < 1.0 and rank0_only():
                logger.warn(
                    "Grad scale less than 1."
                    "  This typically indicates that fp16 training is unstable."
                    "  Consider switching from mixed to autocast or autocast to off"
                )

class WeightTracker(nn.Module):
    """ Weight scaling module for auxiliary tasks. Maintains its own mask (0-1) for each aux task """
    def __init__(self, cfg, device): # aux cfg
        super().__init__()
        self.cfg = cfg
        self.masks = torch.ones(len(cfg.tasks), device=device)
        self.device = device
        self.scale = False
        self.num_calls = 0
        self.step = 0
        if cfg.distribution == "one-hot": # uniform is default, ignored
            self.scale = True
            self.masks *= 0
            self.masks[self.step] = 1

    def forward(self, *x):
        raise NotImplementedError

    def get_scaled_losses(self, losses):
        if self.scale:
            self.num_calls += 1
            step = int(self.num_calls / 4e4) # want to do at 2M frames 8000 * this is actually 500000 ~ 2.5M 8000 ~ 500000
            if self.step != step:
                self.step = step
                if step >= len(self.cfg.tasks):
                    self.scale = False # no more scaling, turn it all on
                    self.masks = torch.ones(len(self.cfg.tasks), device=self.device)
                else:
                    self.masks *= 0
                    self.masks[step] = 1
        return self.masks * torch.stack(losses)
