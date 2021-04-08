#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional
import random
import json
import attr
import contextlib

import numpy as np
import tqdm
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.jit import Final

from habitat import Config, logger
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import (
    observations_to_image,
    save_semantic_frame
)
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.auxiliary_tasks import get_aux_task_classes
from habitat_baselines.rl.ppo.curiosity import ForwardCuriosity
from habitat_baselines.rl.models.rednet import load_rednet
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    batch_obs,
    batch_list,
    generate_video,
    linear_decay,
    is_fp16_autocast_supported,
    is_fp16_supported,
)

from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)

from habitat_baselines.rl.ppo import (
    PPO
)
from habitat_baselines.rl.ppo.encoder_dict import (
    get_vision_encoder_inputs
)

class Diagnostics:
    basic = "basic" # dummy to record episode stats (for t-test)

    actions = "actions"

    gps = "gps"

    heading = "heading"

    weights = "weights"

    top_down_map = "top_down_map"

    episode_info = "episode_info"

    episode_info_full = "episode_info_full"

    # The following three diagnostics are pretty large. Be careful.
    """ internal_activations:
        Records for dynamical analysis. Holds N lists, one per episode.
        Each list is of episode length T, and has:
        - belief hidden states # T x K x H
        - fused hidden state # T x H
        - sensor embeddings # T x H
        - policy logits # T x A
        - critic values # T x 1
    """
    internal_activations = "internal_activations"

    """ observations:
        Sensor observations, pre-embedding. Non-visual inputs only. # key, T x H
    """
    observations = "observations"

    """ observations:
        Sensor observations, pre-embedding, pre-preprocessing. Visual inputs only. # key, THWC
    """
    visual_observations = "visual_observations"

    # Note, we don't record preprocessed visual observations, but we prob don't need them.

    # Following three are typically for probing
    """ d2g:
        Per timestep distance to closest goal (as used in the reward sensor)
    """
    d2g = "d2g"

    """ room_cat:
        Per timestep room category
    """
    room_cat = "room_cat"

    """ visit_count:
        Per timestep current tile visit count, from coverage reward
    """
    visit_count = "visit_count"

    collisions_t = "collisions_t" # collisions per timestep to distinguish debris spawn
    coverage_t = "coverage_t" # coverage per timestep to distinguish debris spawn
    sge_t = "sge_t" # SGE per timestep to distinguish debris spawn

@baseline_registry.register_trainer(name="ppo")
class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.obs_space = None
        self.obs_transforms = []
        if config is not None:
            # logger.info(f"config: {config}")
            self.checkpoint_prefix = config.TENSORBOARD_DIR.split('/')[-1]

        self._static_encoder = False
        self._encoder = None
        self.count_steps = 0

        if self.config.RL.fp16_mode not in ("off", "autocast", "mixed"):
            raise RuntimeError(
                f"Unknown fp16 mode '{self.config.RL.fp16_mode}'"
            )

        if self.config.RL.fp16_mode != "off" and not torch.cuda.is_available():
            logger.warn(
                "FP16 requires CUDA but CUDA is not available, setting to off"
            )

        self._fp16_mixed = self.config.RL.fp16_mode == "mixed"
        self._fp16_autocast = self.config.RL.fp16_mode == "autocast"

        if self._fp16_mixed and not is_fp16_supported():
            raise RuntimeError(
                "FP16 requires PyTorch >= 1.6.0, please update your PyTorch"
            )

        if self._fp16_autocast and not is_fp16_autocast_supported():
            raise RuntimeError(
                "FP16 autocast requires PyTorch >= 1.7.1, please update your PyTorch"
            )

    def _setup_auxiliary_tasks(self, aux_cfg, ppo_cfg, task_cfg, observation_space=None, is_eval=False, policy_encoders=["rgb", "depth"]):
        r"""
            policy_encoders: If an auxiliary sensor is not used for the policy, we will make one
        """
        aux_task_strings = [task.lower() for task in aux_cfg.tasks]
        if "semanticcpca" in aux_task_strings and ppo_cfg.POLICY.USE_SEMANTICS:
            raise Exception("I don't think using a separate semantic cpca task and feeding semantics into our main encoder are compatible")
        # Differentiate instances of tasks by adding letters
        aux_counts = {}
        for i, x in enumerate(aux_task_strings):
            if x in aux_counts:
                aux_task_strings[i] = f"{aux_task_strings[i]}_{aux_counts[x]}"
                aux_counts[x] += 1
            else:
                aux_counts[x] = 1

        logger.info(f"Auxiliary tasks: {aux_task_strings}")

        num_recurrent_memories = 1
        # Currently we have two places for policy name.. not good. Will delete once baselines are run
        if self.config.RL.PPO.policy != "BASELINE":
            raise Exception("I don't think you meant to set this policy")

        policy = baseline_registry.get_policy(ppo_cfg.POLICY.name)

        hidden_sizes = None
        if policy.IS_MULTIPLE_BELIEF:
            proposed_num_beliefs = ppo_cfg.POLICY.BELIEFS.NUM_BELIEFS
            num_recurrent_memories = len(aux_cfg.tasks) if proposed_num_beliefs == -1 else proposed_num_beliefs
            if policy.IS_RECURRENT:
                num_recurrent_memories += 1

        init_aux_tasks = []
        encoder_insts = {}
        if not is_eval:
            task_classes, encoder_classes = get_aux_task_classes(aux_cfg) # supervised is a dict
            for encoder in encoder_classes: # This is a dict of other encoders we want
                if encoder in policy_encoders: # If it already exists
                    pass
                encoder_insts[encoder] = encoder_classes[encoder](observation_space, ppo_cfg.hidden_size).to(self.device)
            for i, task in enumerate(aux_cfg.tasks):
                task_class = task_classes[i]
                # * We previously constructed the extra encoders during aux task setup, but they are best constructed separately beforehand (for attachment to aux task AND policy)
                # * We don't actually need it here, so we disable for now
                # req_sensors = {
                #     name: encoder_insts[name] for name in task_class.get_required_sensors(aux_cfg[task])
                # }
                # Currently the tasks which need a given encoder hold the module itself.
                hidden_size = None # will go to sensible default
                aux_module = task_class(
                    ppo_cfg, aux_cfg[task], task_cfg, self.device, \
                    observation_space=observation_space,
                    # sensor_encoders=req_sensors,
                    hidden_size=hidden_size).to(self.device)
                init_aux_tasks.append(aux_module)

        return init_aux_tasks, num_recurrent_memories, aux_task_strings, encoder_insts

    def _setup_curiosity(self, ppo_cfg, task_cfg, embedding_size):
        return ForwardCuriosity(ppo_cfg, task_cfg, embedding_size)

    def get_ppo_class(self):
        return PPO

    @property
    def obs_space(self):
        if self._obs_space is None and self.envs is not None:
            self._obs_space = self.envs.observation_spaces[0]

        return self._obs_space

    @obs_space.setter
    def obs_space(self, new_obs_space):
        self._obs_space = new_obs_space

    def _setup_actor_critic_agent(
        self,
        ppo_cfg: Config,
        task_cfg: Config,
        aux_cfg: Config = None,
        aux_tasks=[],
        policy_encoders=["rgb", "depth"],
        aux_encoders=None,
    ) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        if len(aux_tasks) != 0 and len(aux_tasks) != len(aux_cfg.tasks):
            raise Exception(f"Policy specifies {len(aux_cfg.tasks)} tasks but {len(aux_tasks)} were initialized.")

        logger.add_filehandler(self.config.LOG_FILE)

        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        self.obs_space = observation_space

        # Default policy settings for object nav
        is_objectnav = "ObjectNav" in task_cfg.TYPE or self.config.MOCK_OBJECTNAV
        additional_sensors = []

        embed_goal = False
        if is_objectnav:
            additional_sensors = ["gps", "compass"]
            embed_goal = True

        # TODO move `ppo_cfg.policy` to `config.RL.POLICY`
        policy = baseline_registry.get_policy(ppo_cfg.POLICY.name)
        self.actor_critic = policy(
            observation_space=self.obs_space,
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            goal_sensor_uuid=task_cfg.GOAL_SENSOR_UUID,
            num_tasks=len(aux_cfg.tasks), # we pass this is in to support eval, where no aux modules are made
            additional_sensors=additional_sensors,
            embed_goal=embed_goal,
            device=self.device,
            config=ppo_cfg.POLICY,
            policy_encoders=policy_encoders,
            num_policy_heads=self._get_policy_head_count(),
            mock_objectnav=self.config.MOCK_OBJECTNAV
        )

        # It's difficult to completely JIT this
        # if policy.IS_JITTABLE and ppo_cfg.POLICY.jit:
        #     self.actor_critic = torch.jit.script(self.actor_critic)

        self.actor_critic.to(self.device)

        curiosity_module = None
        if ppo_cfg.CURIOSITY.USE_CURIOSITY:
            curiosity_module = \
                self._setup_curiosity(ppo_cfg, task_cfg, self.actor_critic.embedding_size)

        if self._fp16_mixed:
            for name, module in self.actor_critic.named_modules():
                if "running_mean_and_var" not in name:
                    module.to(dtype=torch.float16)

        self.agent = self.get_ppo_class()(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            aux_loss_coef=ppo_cfg.aux_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            aux_tasks=aux_tasks,
            aux_cfg=aux_cfg,
            curiosity_cfg=ppo_cfg.CURIOSITY,
            curiosity_module=curiosity_module,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
            aux_encoders=aux_encoders,
            aux_map=ppo_cfg.POLICY.BELIEFS.AUX_MAP, # TODO reroute
            importance_weight=self.config.RL.REWARD_FUSION.SPLIT.IMPORTANCE_WEIGHT,
            fp16_autocast=self._fp16_autocast,
            fp16_mixed=self._fp16_mixed,
        ).to(self.device)

        self.load_pretrained_weights()
        self.agent.script()

        self.semantic_predictor = None
        if self.config.RL.POLICY.TRAIN_PRED_SEMANTICS:
            self.semantic_predictor = load_rednet(
                self.device,
                ckpt=ppo_cfg.POLICY.EVAL_SEMANTICS_CKPT,
                resize=True # since we train on half-vision
            )
            self.semantic_predictor.eval()


    def load_pretrained_weights(self):
        # Load a pre-trained visual encoder. (This is hardcoded to support rgbd loading)
        # Note - this will be overwritten by checkpoints if we're not training from scratch
        if not self.config.RL.PPO.POLICY.pretrained_encoder:
            return
        if self.config.RL.PPO.POLICY.USE_SEMANTICS:
            # Merged with semantic - we can't load separate rgbd weight
            return
        pretrained_state = torch.load(
            self.config.RL.PPO.POLICY.pretrained_weights, map_location="cpu"
        )
        spliced_mean_and_var = {
            k.split(".")[-1]: v for k, v in pretrained_state["state_dict"].items()
            if "running_mean_and_var" in k
        }
        modified_mean_and_var = {
            k: v.view(1, 4, 1, 1)
            for k, v in spliced_mean_and_var.items()
            if "_var" in k or "_mean" in k
        }
        spliced_state = {
            k: v for k, v in pretrained_state["state_dict"].items()
            if "running_mean_and_var" not in k
        }

        # We try twice (with different prefixes) due to some compatibility issues in checkpoints and model versions
        # This first version uses DDPPO weights - in other models, the visual encoder belongs on the "Net" (what we call the core/belief)
        ve_str = 'actor_critic.net.visual_encoder.'
        visual_dict = {
            k[len(ve_str):]: v
            for k, v in spliced_state.items()
            if k.startswith(ve_str)
        }
        rgbd_module = self.actor_critic.visual_encoders.encoders["['depth', 'rgb']"][0]
        if len(visual_dict) > 0:
            rgbd_module.load_state_dict(visual_dict)
            return
        # This second version is for when we load with our own weights - where the visual encoder belongs to the policy (as a shared base)
        ve_str = 'actor_critic.visual_encoder.'
        visual_dict = {
            k[len(ve_str):]: v
            for k, v in spliced_state.items()
            if k.startswith(ve_str)
        }
        rgbd_module.load_state_dict(visual_dict)
        self.actor_critic.running_mean_and_var.load_state_dict(modified_mean_and_var)

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        def _cast(t: torch.Tensor):
            if t.dtype == torch.float16:
                return t.to(dtype=torch.float32)
            else:
                return t
        checkpoint = {
            "state_dict": {
                k: _cast(v) for k, v in self.agent.state_dict().items()
            },
            # FIXME optim state, should I cast it?
            "optim_state": self.agent.optim_state_dict(),
            # "optim_state": {
            #     k: _cast(v) for k, v in self.agent.optimizer.state_dict().items()
            # },
            # "state_dict": self.agent.state_dict(),
            # "optim_state": self.agent.optimizer.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _get_policy_head_count(self):
        reward_keys = self.config.RL.POLICIES
        if reward_keys[0] == "none" and len(reward_keys) == 1:
            return 1
        if self.config.RL.REWARD_FUSION.STRATEGY == "SPLIT":
            return 2
        return 1

    def _build_rewards(
        self,
        env_rewards,
        metrics,
    ):
        r"""
            In order to support more complex reward operations, we treat rewards as normal metrics.
            The env still returns rewards as per gym API, but env reward should be combined with measures as configured.
            Typically, the env reward will just contain slack.
            Args:
                env_rewards: [b] env reward
                metrics: dict of [b] (reward) measures.
                Note these sizes don't have extra feature dims since rewards are expected to be scalars.
            Return:
                env_rewards: k x b, where k is the number of policy heads (typically 1)
        """
        # extract the reward metrics
        reward_keys = self.config.RL.POLICIES
        if reward_keys[0] == "none" and len(reward_keys) == 1:
            return env_rewards.unsqueeze(0)
        strategy = self.config.RL.REWARD_FUSION.STRATEGY
        if strategy == "SUM":
            return (env_rewards + sum(metrics[p] for p in reward_keys)).unsqueeze(0)
        reward_a = sum(metrics[p] for p in reward_keys[:-1]) + env_rewards
        reward_b = metrics[reward_keys[-1]]
        if self.config.RL.REWARD_FUSION.ENV_ON_ALL:
            reward_b = reward_b + env_rewards
        if self.config.RL.REWARD_FUSION.STRATEGY == "RAMP":
            # Ramps from a to b
            ramp_factor = min(1, max(0, (
                self.count_steps - self.config.RL.REWARD_FUSION.RAMP.START
            ) / (
                self.config.RL.REWARD_FUSION.RAMP.END - self.config.RL.REWARD_FUSION.RAMP.START
            )))
            return (reward_a * (1 - ramp_factor) + reward_b * ramp_factor).unsqueeze(0)
        elif self.config.RL.REWARD_FUSION.STRATEGY == "SPLIT":
            return torch.stack([reward_a, reward_b], dim=0)
        raise NotImplementedError

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, current_episode_env_reward, running_episode_stats, prior_obs_state=None
    ):
        pth_time = 0.0
        env_time = 0.0

        ppo_cfg = self.config.RL.PPO
        curiosity_cfg = ppo_cfg.CURIOSITY

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad(), torch.cuda.amp.autocast() if self._fp16_autocast else contextlib.suppress():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            behavioral_index = 0
            if self._get_policy_head_count() > 1 and self.count_steps > self.config.RL.REWARD_FUSION.SPLIT.TRANSITION:
                behavioral_index = 1

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
                obs,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.get_recurrent_states()[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
                return_features=True,
                behavioral_index=behavioral_index
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        env_time += time.time() - t_step_env
        t_update_stats = time.time()

        # Hardcoded
        def map_to_full_metric(m):
            if m == 'reached':
                return ['coverage', 'reached']
            elif m == 'visit_count':
                return ['coverage', 'visit_count']
            elif m == "mini_reached":
                return ['coverage', 'mini_reached']
            else:
                return [m]
        TRACKED_METRICS = [map_to_full_metric(m) for m in self.config.RL.PPO.ROLLOUT.METRICS]
        tracked_metrics = batch_list(infos, device=self.device, whitelist=TRACKED_METRICS)

        batch = batch_obs(observations, device=self.device)
        if self.semantic_predictor is not None:
            batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
            if ppo_cfg.POLICY.EVAL_SEMANTICS_CKPT == "/srv/share/jye72/rednet/rednet_semmap_mp3d_40.pth":
                batch["semantic"] -= 1
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rewards = torch.tensor(
            rewards, dtype=torch.float, device=current_episode_reward.device
        )
        POLICY_METRICS = [map_to_full_metric(m) for m in self.config.RL.POLICIES if m is not "none"] # Careful not to duplicate this
        policy_metrics = batch_list(infos, device=rewards.device, whitelist=POLICY_METRICS)
        rewards = self._build_rewards(rewards, policy_metrics)
        rewards = rewards.unsqueeze(-1) # b x k -> b x k x 1
        # reward [k x b x 1] * masks [b x 1] -> [k x b x 1]

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )
        current_episode_env_reward += rewards

        curiosity_obs = None
        if curiosity_cfg.USE_CURIOSITY:
            # ! Curiosity not supported for multi-rewards. Assuming bonus belongs to first dimension
            curiosity_obs = obs[curiosity_cfg.VISION_KEY]
            if prior_obs_state is not None:
                with torch.no_grad():
                    # Pass in the state after seeing the prior observation (our input state)
                    prior_state = rollouts.get_recurrent_states()[rollouts.step] if curiosity_cfg.USE_BELIEF else None
                    fp_error = self.agent.get_curiosity_error(
                        prior_obs_state,
                        curiosity_obs,
                        rollouts.prev_actions[rollouts.step],
                        beliefs=prior_state
                    )
                    curiosity_reward = torch.log(fp_error + 1.0).unsqueeze(1).to(rewards.device) * curiosity_cfg.REWARD_SCALE
                    # If the episode has ended (mask is 0), prev and current obs are not in same scene, zero reward
                    curiosity_reward = curiosity_reward * masks # b x 1
                    rewards[:,0] = rewards[:, 0] + curiosity_reward

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward # only add reward at episode end?
        running_episode_stats["env_reward"] += (1 - masks) * current_episode_env_reward
        running_episode_stats["count"] += 1 - masks
        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v

        current_episode_reward *= masks
        current_episode_env_reward *= masks

        if self._static_encoder:
            if self._fp16_mixed:
                raise Exception("Not implemented")
            with torch.no_grad(), torch.cuda.amp.autocast() if self._fp16_autocast else contextlib.suppress():
                batch["visual_features"] = self._encoder(batch)

        if self._get_policy_head_count() == 1: # Single-policy agents don't return the policy dimension.
            values = values.unsqueeze(1)
            actions_log_probs = actions_log_probs.unsqueeze(1)

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs, # b x k x 1
            values, # b x k x 1
            rewards, # k x b x 1
            masks,
            tracked_metrics
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs, curiosity_obs

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast() if self._fp16_autocast else contextlib.suppress():
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.get_recurrent_states()[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach()

        behavioral_index = 0
        if self._get_policy_head_count() > 1 and self.count_steps > self.config.RL.REWARD_FUSION.SPLIT.TRANSITION:
            behavioral_index = 1

        iw_clipped = ppo_cfg.SPLIT_IW_BOUNDS if hasattr(ppo_cfg, 'SPLIT_IW_BOUNDS') else \
            [1.0 - ppo_cfg.clip_param, 1.0 + ppo_cfg.clip_param]
        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau,
            behavioral_index=behavioral_index,
            importance_weight=self.config.RL.REWARD_FUSION.SPLIT.IMPORTANCE_WEIGHT,
            weight_clip=iw_clipped,
        )

        (
            value_loss,
            action_loss,
            dist_entropy,
            aux_task_losses,
            aux_dist_entropy,
            aux_weights,
            inv_curiosity,
            fwd_curiosity,
        ) = self.agent.update(
            rollouts,
            ppo_cfg.gamma,
            behavioral_index=behavioral_index
        )

        rollouts.after_update()
        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
            aux_task_losses,
            aux_dist_entropy,
            aux_weights,
            inv_curiosity,
            fwd_curiosity,
        )

    def _make_deltas(self, window_episode_stats):
        deltas = {
            k: (
                (v[-1] - v[0]).flatten(start_dim=-2).sum(dim=-1) # k x b x 1 OR b x 1 -> k or 1
                if len(v) > 1
                else v[0].flatten(start_dim=-2).sum(dim=-1)
            )
            for k, v in window_episode_stats.items()
        }
        # Get items, and flatten rewards to report multi-policy
        flat_deltas = {}
        for k, v in deltas.items():
            if len(v.size()) > 0:
                flat_deltas[k] = v[0].item()
                for i in range(1, v.size(0)):
                    flat_deltas[f"{k}_{i}"] = v[i].item()
            else:
                flat_deltas[k] = v.item()

        flat_deltas["count"] = max(flat_deltas["count"], 1.0)
        return flat_deltas

    def train(self, ckpt_path="", ckpt=-1, start_updates=0) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        ppo_cfg = self.config.RL.PPO
        task_cfg = self.config.TASK_CONFIG.TASK

        policy_encoders_map = get_vision_encoder_inputs(ppo_cfg)

        """
        Initialize auxiliary tasks
        """
        aux_cfg = self.config.RL.AUX_TASKS

        init_aux_tasks, num_recurrent_memories, aux_task_strings, aux_encoder_insts = \
            self._setup_auxiliary_tasks(aux_cfg, ppo_cfg, task_cfg,
                observation_space=observation_space, policy_encoders=policy_encoders_map)

        self._setup_actor_critic_agent(
            ppo_cfg, task_cfg, aux_cfg,
            init_aux_tasks,
            aux_encoders=aux_encoder_insts,
            policy_encoders=policy_encoders_map
        )

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.obs_space,
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
            num_recurrent_memories=num_recurrent_memories,
            num_policy_heads=self._get_policy_head_count(),
            metrics=ppo_cfg.ROLLOUT.METRICS
        )

        rollouts.to(self.device)
        if self._fp16_mixed:
            rollouts.to_fp16()

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        if self.semantic_predictor is not None:
            batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.actor_critic.parameters())
            )
        )
        logger.info(
            "all parameters: {}".format(
                sum(param.numel() for param in self.agent.get_parameters())
            )
        )

        reward_count = self._get_policy_head_count()
        current_episode_env_reward = torch.zeros(reward_count, self.envs.num_envs, 1) # num policies x envs x 1? (last dim is just a quirk, I think)
        current_episode_reward = torch.zeros(reward_count, self.envs.num_envs, 1) # Include intrinsic rewards
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(reward_count, self.envs.num_envs, 1),
            env_reward=torch.zeros(reward_count, self.envs.num_envs, 1)
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        self.count_steps = 0
        elapsed_steps = 0
        count_checkpoints = 0
        if ckpt != -1:
            logger.info(
                f"Resuming runs at checkpoint {ckpt}. Timing statistics are not tracked properly."
            )
            assert ppo_cfg.use_linear_lr_decay is False and ppo_cfg.use_linear_clip_decay is False, "Resuming with decay not supported"
            count_checkpoints = ckpt + 1
            self.count_steps = start_updates * ppo_cfg.num_steps * self.config.NUM_PROCESSES # default estimate
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            # ! We may be changing the architecture, thus we sometimes load checkpoints without all the weights we need.
            is_warm_start = ckpt_path == self.config.RL.POLICY.PRETRAINED_CKPT
            self.agent.load_state_dict(ckpt_dict["state_dict"], strict=not is_warm_start)
            if "optim_state" in ckpt_dict:
                self.agent.load_optim_state(ckpt_dict["optim_state"], is_warm_start=is_warm_start)
            else:
                logger.warn("No optimizer state loaded, results may be funky")
            if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
                self.count_steps = ckpt_dict["extra_state"]["step"]

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR,
            flush_secs=self.flush_secs,
            purge_step=self.count_steps,
        ) as writer:

            for update in range(start_updates, self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                prior_obs_state = None # For curiosity

                for step in range(ppo_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                        prior_obs_state
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_reward, current_episode_env_reward, running_episode_stats, prior_obs_state=prior_obs_state
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    self.count_steps += delta_steps
                    elapsed_steps += delta_steps

                (
                    delta_pth_time,
                    value_losses,
                    action_losses,
                    dist_entropy,
                    aux_task_losses,
                    aux_dist_entropy,
                    aux_weights,
                    inv_curiosity,
                    fwd_curiosity
                ) = self._update_agent(ppo_cfg, rollouts)
                pth_time += delta_pth_time

                for k, v in running_episode_stats.items():
                    window_episode_stats[k].append(v.clone())

                deltas = self._make_deltas(window_episode_stats)

                self.report_train_metrics(writer, {
                    "aux_entropy": aux_dist_entropy,
                    "inv_curiosity_loss": inv_curiosity,
                    "fwd_curiosity_loss": fwd_curiosity,
                },
                    deltas, dist_entropy, [value_losses, action_losses], aux_task_losses, self.count_steps, elapsed_steps, update,
                    env_time, pth_time, t_start, window_episode_stats,
                    aux_weights, aux_task_strings)

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"{self.checkpoint_prefix}.{count_checkpoints}.pth", dict(step=self.count_steps)
                    )
                    count_checkpoints += 1

        self.envs.close()

    def project_out(
        self, states, projection_path='/srv/share/jye72/base-full_timesteps.pth'
    ):
        r"""
        # states l x b x k x h
        Project states out of the dimension which has vectors.
        used in experimentation with the time dimension to see if we can stop early stops/chaotic behavior.
        """
        if self._projections is None:
            axes = torch.load(projection_path, map_location=self.device).float() # k x h, see probes.py
            intercepts = axes[:, -1]
            axes = axes[:, :-1]
            norms = axes.norm(dim=1) # k
            norm_axes = axes / axes.norm(dim=1, keepdim=True) # k x h
            self._project_in = ((self.config.EVAL.PROJECT_OUT - intercepts)/ norms).unsqueeze(1) * norm_axes  # k x 1 * k x h -> k x h
            # https://statisticaloddsandends.wordpress.com/2018/02/02/projection-matrix-for-a-1-dimensional-subspace/
            projection_matrices = []
            for axis in norm_axes: # h
                projection_matrices.append(torch.outer(axis, axis.T))
            self._projections = torch.stack(projection_matrices, dim=0).float() # k x h x h

        projected_states = []
        for i in range(0, states.size(-2)):
            # states[0] is just the layer of the RNN, we use a 1 layer GRU.
            projected = torch.matmul(self._projections[i].unsqueeze(0), states[0, :, i].float().unsqueeze(2)).squeeze(-1) # 1 x h x h @ b x h x 1
            # b x h
            project_out = states[0, :, i] - projected
            projected_states.append(project_out) # b x h
        project_out = torch.stack(projected_states, dim=1)
        states[0] = project_out + self._project_in # b x k x h + k x h


        return states

    @torch.no_grad()
    def _simple_eval(
        self,
        ckpt_dict: dict,
        config: Config
    ):
        # Match EvalAI docker while still mostly following default eval structure (parity with local eval is proven up to 300 episodes)
        # * this was originally written trying to identify mismatch between local val and evalai test-std.
        # * no bug was found; the issue is likely distribution shift.

        # Config
        aux_cfg = config.RL.AUX_TASKS
        ppo_cfg = config.RL.PPO
        task_cfg = config.TASK_CONFIG.TASK

        # Load spaces (via env)
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))

        # ! Agent setup
        policy_encoders = get_vision_encoder_inputs(ppo_cfg)

        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        is_objectnav = "ObjectNav" in task_cfg.TYPE or self.config.MOCK_OBJECTNAV
        additional_sensors = []
        embed_goal = False
        if is_objectnav:
            additional_sensors = ["gps", "compass"]
            embed_goal = True

        def _get_policy_head_count(config):
            reward_keys = config.RL.POLICIES
            if reward_keys[0] == "none" and len(reward_keys) == 1:
                return 1
            if config.RL.REWARD_FUSION.STRATEGY == "SPLIT":
                return 2
            return 1

        policy_class = baseline_registry.get_policy(ppo_cfg.POLICY.name)
        self.actor_critic = policy_class(
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            goal_sensor_uuid=task_cfg.GOAL_SENSOR_UUID,
            num_tasks=len(aux_cfg.tasks), # we pass this is in to support eval, where no aux modules are made
            additional_sensors=additional_sensors,
            embed_goal=embed_goal,
            device=self.device,
            config=ppo_cfg.POLICY,
            policy_encoders=policy_encoders,
            num_policy_heads=_get_policy_head_count(config),
            mock_objectnav=config.MOCK_OBJECTNAV
        ).to(self.device)

        self.num_recurrent_memories = self.actor_critic.net.num_tasks
        if self.actor_critic.IS_MULTIPLE_BELIEF:
            proposed_num_beliefs = ppo_cfg.POLICY.BELIEFS.NUM_BELIEFS
            self.num_recurrent_memories = len(aux_cfg.tasks) if proposed_num_beliefs == -1 else proposed_num_beliefs
            if self.actor_critic.IS_RECURRENT:
                self.num_recurrent_memories += 1

        self.actor_critic.load_state_dict(
            {
                k.replace("actor_critic.", ""): v
                for k, v in ckpt_dict["state_dict"].items()
                if "actor_critic" in k
            }
        )
        self.actor_critic.eval()

        self.semantic_predictor = None
        if ppo_cfg.POLICY.USE_SEMANTICS:
            self.semantic_predictor = load_rednet(
                self.device,
                ckpt=ppo_cfg.POLICY.EVAL_SEMANTICS_CKPT,
                resize=True # since we train on half-vision
            )
            self.semantic_predictor.eval()

        self.behavioral_index = 0
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            count_steps = ckpt_dict["extra_state"]["step"]
            if _get_policy_head_count(config) > 1 and count_steps > config.RL.REWARD_FUSION.SPLIT.TRANSITION:
                self.behavioral_index = 1

        # Load other items
        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            self.num_recurrent_memories,
            ppo_cfg.hidden_size,
            device=self.device,
        )

        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.bool
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )


        # * Do eval
        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    f", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device) # Note docker appears to get a single observation as opposed to a list (1 proc)
        if self.semantic_predictor is not None:
            batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        stats_episodes = dict()  # dict of dicts that stores stats per episode
        total_stats = []
        dones_per_ep = dict()

        pbar = tqdm.tqdm(total=number_of_eval_episodes)

        self.step = 0
        self.ep = 0

        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()
            with torch.no_grad(), torch.cuda.amp.autocast() if self._fp16_autocast else contextlib.suppress():

                # Match EvalAI settings
                if config.EVAL.restrict_gps:
                    batch["gps"][:,1] = 0

                deterministic = hasattr(self.config.EVAL, "DETERMINISTIC") and self.config.EVAL.DETERMINISTIC
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                    *_
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=deterministic,
                    behavioral_index=self.behavioral_index,
                )
                prev_actions.copy_(actions)

                if self.config.EVAL.PROJECT_OUT >= 0:
                    test_recurrent_hidden_states = self.project_out(test_recurrent_hidden_states, projection_path=self.config.EVAL.PROJECTION_PATH)

            self.step += 1

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)
            if self.semantic_predictor is not None:
                batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            not_done_masks = torch.tensor(
                [[False] if done else [True] for done in dones],
                dtype=torch.bool,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                next_k = (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                )
                if dones_per_ep.get(next_k, 0) == 1:
                    envs_to_pause.append(i) # wait for the rest

                if not_done_masks[i].item() == 0:
                    episode_stats = dict()

                    episode_stats["reward"] = current_episode_reward[i].item()
                    current_episode_reward[i] = 0

                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )

                    # use scene_id + episode_id as unique id for storing stats
                    k = (
                        current_episodes[i].scene_id,
                        current_episodes[i].episode_id,
                    )
                    dones_per_ep[k] = dones_per_ep.get(k, 0) + 1

                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                            dones_per_ep[k],
                        )
                    ] = episode_stats

                    pbar.update()

                    print(f'{self.ep} reset {self.step}')
                    self.step = 0
                    self.ep += 1

                # episode continues
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                _,
                _,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                {},
                _,
            )

        # Report results
        num_episodes = len(stats_episodes)
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.8f}")

        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]
            logger.info(f"\n Step ID (update): {step_id}")
        self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
        log_diagnostics=[],
        output_dir='.',
        label='.',
        num_eval_runs=1,
        skip_log=False,
        simple_eval=False,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        self._projections = None
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        # Config
        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()
        aux_cfg = config.RL.AUX_TASKS
        ppo_cfg = config.RL.PPO
        task_cfg = config.TASK_CONFIG.TASK

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if simple_eval:
            self._simple_eval(ckpt_dict, config)
            return

        # Add additional measurements
        config.defrost()
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
        if len(self.config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        config.freeze()

        # Load spaces (via env)
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))

        # ! Agent setup
        policy_encoders = get_vision_encoder_inputs(ppo_cfg)

        # pass in aux config if we're doing attention
        self._setup_actor_critic_agent(
            ppo_cfg, task_cfg, aux_cfg,
            policy_encoders=policy_encoders
        )

        self.actor_critic = self.agent.actor_critic # We don't use PPO info
        self.actor_critic.load_state_dict(
            {
                k.replace("actor_critic.", ""): v
                for k, v in ckpt_dict["state_dict"].items()
                if "actor_critic" in k
            }
        )
        self.actor_critic.eval()

        logger.info(
            "agent number of trainable parameters: {}".format(
                sum(
                    param.numel()
                    for param in self.agent.parameters()
                    if param.requires_grad
                )
            )
        )

        self.semantic_predictor = None
        if ppo_cfg.POLICY.USE_SEMANTICS and not ppo_cfg.POLICY.EVAL_GT_SEMANTICS:
            self.semantic_predictor = load_rednet(
                self.device, ckpt=ppo_cfg.POLICY.EVAL_SEMANTICS_CKPT, resize=True, # since we train on half-vision
                stabilize=ppo_cfg.POLICY.EVAL_SEMANTICS_STABILIZE
                # ! TODO sub no resize back in, rn it's a no-op
                # self.device, ckpt=ppo_cfg.POLICY.EVAL_SEMANTICS_CKPT, resize=not self.config.RL.POLICY.FULL_VISION
            )
            self.semantic_predictor.eval()

        self.behavioral_index = 0
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            self.count_steps = ckpt_dict["extra_state"]["step"]
        if self._get_policy_head_count() > 1 and self.count_steps > self.config.RL.REWARD_FUSION.SPLIT.TRANSITION:
            self.behavioral_index = 1

        # Load other items
        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        if self.actor_critic.IS_MULTIPLE_BELIEF:
            # ! This can be skipped once we have belief specification
            _, num_recurrent_memories, _, _ = self._setup_auxiliary_tasks(aux_cfg, ppo_cfg, task_cfg, is_eval=True)
            test_recurrent_hidden_states = test_recurrent_hidden_states.unsqueeze(2).repeat(1, 1, num_recurrent_memories, 1)

        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.bool
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    f", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        if self.semantic_predictor is not None:
            # batch["gt_semantic"] = batch["semantic"]
            batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
            if ppo_cfg.POLICY.EVAL_SEMANTICS_CKPT == "/srv/share/jye72/rednet/rednet_semmap_mp3d_40.pth":
                batch["semantic"] -= 1

        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        stats_episodes = dict()  # dict of dicts that stores stats per episode
        total_stats = []
        dones_per_ep = dict()

        # Video and logging
        aux_task_strings = self.config.RL.AUX_TASKS.tasks
        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]

        is_full_eval = len(log_diagnostics) > 0 # len(self.config.VIDEO_OPTION) == 0 and
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        video_indices = range(self.config.TEST_EPISODE_COUNT)
        print(f"Videos: {video_indices}")

        # Logging more extensive evaluation stats for analysis
        per_timestep_diagnostics = [d for d in log_diagnostics if d in [
            Diagnostics.actions, Diagnostics.gps, Diagnostics.heading,
            Diagnostics.weights, Diagnostics.internal_activations,
            Diagnostics.observations, Diagnostics.visual_observations,
            Diagnostics.room_cat, Diagnostics.d2g, Diagnostics.visit_count,
            Diagnostics.coverage_t, Diagnostics.collisions_t, Diagnostics.sge_t
        ]]
        d_stats = {}
        if len(per_timestep_diagnostics) > 0:
            for d in per_timestep_diagnostics:
                d_stats[d] = [
                    [] for _ in range(self.config.NUM_PROCESSES)
                ] # stored as nested list envs x timesteps x k (# tasks)

        pbar = tqdm.tqdm(total=number_of_eval_episodes * num_eval_runs)

        while (
            len(stats_episodes) < number_of_eval_episodes * num_eval_runs
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()
            with torch.no_grad(), torch.cuda.amp.autocast() if self._fp16_autocast else contextlib.suppress():
                weights_output = None
                if (len(self.config.VIDEO_OPTION) > 0 or Diagnostics.weights in log_diagnostics) and \
                    self.actor_critic.IS_MULTIPLE_BELIEF and self.actor_critic.LATE_FUSION:
                    num_modules = ppo_cfg.POLICY.BELIEFS.NUM_BELIEFS
                    if num_modules == -1:
                        num_modules = len(aux_task_strings)
                    aux_task_strings = aux_task_strings[:num_modules]
                    weights_output = torch.empty(self.envs.num_envs, num_modules)

                # Match EvalAI settings
                if config.EVAL.restrict_gps:
                    batch["gps"][:,1] = 0

                deterministic = hasattr(self.config.EVAL, "DETERMINISTIC") and self.config.EVAL.DETERMINISTIC
                (
                    value,
                    actions,
                    action_log_probs,
                    test_recurrent_hidden_states,
                    *other_outputs
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=deterministic,
                    weights_output=weights_output,
                    behavioral_index=self.behavioral_index,
                    return_all_activations=Diagnostics.internal_activations in log_diagnostics,
                )
                prev_actions.copy_(actions)

                if self.config.EVAL.PROJECT_OUT >= 0:
                    test_recurrent_hidden_states = self.project_out(test_recurrent_hidden_states, self.config.EVAL.PROJECTION_PATH)

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            if len(log_diagnostics) > 0:
                for i in range(self.envs.num_envs):
                    if Diagnostics.actions in log_diagnostics:
                        d_stats[Diagnostics.actions][i].append(prev_actions[i].item())
                    if Diagnostics.weights in log_diagnostics:
                        aux_weights = None if weights_output is None else weights_output[i]
                        if aux_weights is not None:
                            d_stats[Diagnostics.weights][i].append(aux_weights.half().tolist())
                    if Diagnostics.internal_activations in log_diagnostics:
                        fused_features, fused_sensors, logits = other_outputs
                        d_stats[Diagnostics.internal_activations][i].append({
                            "beliefs": test_recurrent_hidden_states[-1, i].half().cpu(),
                            "fused_belief": fused_features[i].half().cpu(),
                            "fused_obs": fused_sensors[i, 0].half().cpu(), # b k h -> h
                            "action_logits": logits[i].half().cpu(),
                            "critic_values": value[i].half().cpu()
                        })
                    if Diagnostics.observations in log_diagnostics:
                        d_stats[Diagnostics.observations][i].append({
                            key: batch[key][i].cpu() for key in ['compass', 'gps'] # [H]
                        })
                    if Diagnostics.visual_observations in log_diagnostics:
                        d_stats[Diagnostics.visual_observations][i].append({
                            key: batch[key][i].cpu() for key in ['rgb', 'depth', 'semantic'] # HWC
                        })
                    if Diagnostics.sge_t in log_diagnostics:
                        d_stats[Diagnostics.sge_t][i].append(infos[i]['goal_vis'])
                    if Diagnostics.collisions_t in log_diagnostics:
                        d_stats[Diagnostics.collisions_t][i].append(infos[i]['collisions']['count'])
                    if Diagnostics.coverage_t in log_diagnostics:
                        d_stats[Diagnostics.coverage_t][i].append({
                            'mini_reached': infos[i]['coverage']['mini_reached'],
                            'reached': infos[i]['coverage']['reached'],
                        })
                    if Diagnostics.visit_count in log_diagnostics:
                        d_stats[Diagnostics.visit_count][i].append(infos[i]['coverage']['visit_count'])
                    if Diagnostics.room_cat in log_diagnostics:
                        d_stats[Diagnostics.room_cat][i].append({
                            'room_cat': infos[i]['region_level']['room_cat'],
                        })
                    if Diagnostics.d2g in log_diagnostics:
                        d_stats[Diagnostics.d2g][i].append(infos[i]['distance_to_goal'])

            batch = batch_obs(observations, device=self.device)
            if self.semantic_predictor is not None:
                # batch["gt_semantic"] = batch["semantic"]
                batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
                if ppo_cfg.POLICY.EVAL_SEMANTICS_CKPT == "/srv/share/jye72/rednet/rednet_semmap_mp3d_40.pth":
                    batch["semantic"] -= 1
                if len(self.config.VIDEO_OPTION) > 0:
                    for i in range(batch["semantic"].size(0)):
                        # observations[i]['gt_semantic'] = observations[i]['semantic']
                        observations[i]['semantic'] = batch["semantic"][i].cpu().numpy()
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            not_done_masks = torch.tensor(
                [[False] if done else [True] for done in dones],
                dtype=torch.bool,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                next_k = (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                )
                if dones_per_ep.get(next_k, 0) == num_eval_runs:
                    envs_to_pause.append(i) # wait for the rest

                if not_done_masks[i].item() == 0:
                    episode_stats = dict()

                    episode_stats["reward"] = current_episode_reward[i].item()
                    current_episode_reward[i] = 0

                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )

                    # use scene_id + episode_id as unique id for storing stats
                    k = (
                        current_episodes[i].scene_id,
                        current_episodes[i].episode_id,
                    )
                    dones_per_ep[k] = dones_per_ep.get(k, 0) + 1

                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                            dones_per_ep[k],
                        )
                    ] = episode_stats

                    if dones_per_ep.get(k, 0) == 1 and len(self.config.VIDEO_OPTION) > 0 and len(stats_episodes) in video_indices:
                        logger.info(f"Generating video {len(stats_episodes)}")
                        category = getattr(current_episodes[i], "object_category", "")
                        if category != "":
                            category += "_"
                        try:
                            if checkpoint_index == -1:
                                ckpt_file = checkpoint_path.split('/')[-1]
                                split_info = ckpt_file.split('.')
                                checkpoint_index = split_info[1]
                            proj_stem = self.config.EVAL.PROJECTION_PATH.split('/')[-1].split('_')[-2]
                            proj_str = f"proj-{proj_stem}-{self.config.EVAL.PROJECT_OUT}" if self.config.EVAL.PROJECT_OUT >= 0 else ""
                            generate_video(
                                video_option=self.config.VIDEO_OPTION,
                                video_dir=self.config.VIDEO_DIR,
                                images=rgb_frames[i],
                                episode_id=current_episodes[i].episode_id,
                                checkpoint_idx=checkpoint_index,
                                metrics=self._extract_scalars_from_info(infos[i]),
                                tag=f"{proj_str}{category}{label}_{current_episodes[i].scene_id.split('/')[-1]}",
                                tb_writer=writer,
                            )
                        except Exception as e:
                            logger.warning(str(e))
                    rgb_frames[i] = []

                    if len(log_diagnostics) > 0:
                        diagnostic_info = dict()
                        for metric in per_timestep_diagnostics:
                            if isinstance(d_stats[metric][i][0], dict):
                                diagnostic_info[metric] = batch_obs(d_stats[metric][i], dtype=torch.half)
                            else:
                                diagnostic_info[metric] = torch.tensor(d_stats[metric][i])
                            d_stats[metric][i] = []

                        # TODO We want to stack this too
                        if Diagnostics.top_down_map in log_diagnostics:
                            top_down_map = infos[i]["top_down_map"]["map"]
                            top_down_map = maps.colorize_topdown_map(
                                top_down_map, fog_of_war_mask=None
                            )
                            diagnostic_info.update(dict(top_down_map=top_down_map))
                        if Diagnostics.episode_info in log_diagnostics:
                            ep_info = attr.asdict(current_episodes[i])
                            if Diagnostics.episode_info_full not in log_diagnostics:
                                del ep_info['goals']
                                del ep_info['shortest_paths']
                                del ep_info['_shortest_path_cache']
                            diagnostic_info.update(dict(
                                episode_info=ep_info,
                            ))

                        total_stats.append(
                            dict(
                                stats=episode_stats,
                                did_stop=bool(prev_actions[i] == 0),
                                info=diagnostic_info,
                            )
                        )
                    pbar.update()

                # episode continues
                else:
                    if len(self.config.VIDEO_OPTION) > 0:
                        aux_weights = None if weights_output is None else weights_output[i]
                        frame = observations_to_image(observations[i], infos[i], current_episode_reward[i].item(), aux_weights, aux_task_strings)
                        rgb_frames[i].append(frame)
                    if Diagnostics.gps in log_diagnostics:
                        d_stats[Diagnostics.gps][i].append(observations[i]["gps"].tolist())
                    if Diagnostics.heading in log_diagnostics:
                        d_stats[Diagnostics.heading][i].append(observations[i]["heading"].tolist())

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                d_stats,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                d_stats,
                rgb_frames,
            )

        # Report results
        num_episodes = len(stats_episodes)
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]
            logger.info(f"\n Step ID (update): {step_id}")
        if label != "train" and num_episodes == 2184 and not skip_log:
            writer.add_scalars(
                "eval_reward",
                {"average reward": aggregated_stats["reward"]},
                step_id,
            )
            metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
            if len(metrics) > 0:
                writer.add_scalars("eval_metrics", metrics, step_id)
                logger.info("eval_metrics")
                logger.info(metrics)
        if len(log_diagnostics) > 0:
            proj_str = f"proj-{self.config.EVAL.PROJECT_OUT}" if self.config.EVAL.PROJECT_OUT >= 0 else ""
            os.makedirs(output_dir, exist_ok=True)
            if Diagnostics.top_down_map in log_diagnostics:
                torch.save(total_stats, os.path.join(output_dir, f'{label}.pth'))
            else:
                meta_stats = {
                    'step_id': step_id,
                    'payload': total_stats
                }
                torch.save(meta_stats, os.path.join(output_dir, f'{proj_str}{label}.pth'))
        self.envs.close()


    def report_train_metrics(
        self,
        writer,
        stats,
        deltas,
        entropy: torch.tensor,
        losses: List[torch.tensor],
        aux_losses,
        count_steps,
        elapsed_steps,
        update,
        env_time,
        pth_time,
        t_start,
        window_episode_stats,
        aux_weights,
        aux_task_strings
    ):
        r"""
            Add stats (torch values that we're too lazy to aggregate),
            Add losses
            Add deltas (stats properly averaged across episodes)
            To TB + logger.
            Extracted since DDPPO trainer has the same code.
            args:
                deltas: dictionary of scalars
        """
        for stat_key, stat_val in stats.items():
            writer.add_scalar(stat_key, stat_val, count_steps)
        # Check for other metrics that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"count"}
        }
        if len(entropy.size()) > 0:
            strs = ["entropy"] + [f"entropy_{i}" for i in range(1, entropy.size(0))]
            for s, e in zip(strs, entropy):
                metrics[s] = e.item()
        else:
            metrics["entropy"] = entropy.item()
        if len(metrics) > 0:
            writer.add_scalars("metrics", metrics, count_steps)

        value_losses = losses[0]
        policy_losses = losses[1]
        losses_strs = ["value", "policy"]
        if len(value_losses.size()) > 0:
            for i in range(1, value_losses.size(0)):
                losses_strs.extend([f"value_{i}", f"policy_{i}"])
        losses = [val.item() for pair in zip(value_losses, policy_losses) for val in pair] + aux_losses
        losses_strs.extend(aux_task_strings)

        writer.add_scalars(
            "losses",
            {k: l for l, k in zip(losses, losses_strs)},
            count_steps,
        )

        if aux_weights is not None:
            writer.add_scalars(
                "aux_weights",
                {k: l for l, k in zip(aux_weights, aux_task_strings)},
                count_steps,
            )

        # Log stats
        if update > 0 and update % self.config.LOG_INTERVAL == 0:
            formatted_losses = [f"{s}: {l:.3g}" for s, l in zip(losses_strs, losses)]
            logger.info(
                "update: {}\t {} \t aux_entropy {:.3g}\t inv curious {:.3g} fwd curious {:.3g}".format(
                    update, formatted_losses, stats["aux_entropy"], stats["inv_curiosity_loss"], stats["fwd_curiosity_loss"]
                )
            )
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    update, elapsed_steps / (time.time() - t_start)
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    update, env_time, pth_time, count_steps
                )
            )

            logger.info(
                "Average window size: {}  {}".format(
                    len(window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )
