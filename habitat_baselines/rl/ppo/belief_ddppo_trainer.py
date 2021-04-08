#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from collections import OrderedDict, defaultdict, deque

import time
import random

import numpy as np
import torch
import torch.distributed as distrib
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    batch_obs,
    linear_decay,
)
from habitat_baselines.rl.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from habitat_baselines.rl.ddppo.algo.ddppo import DDPPO
from habitat_baselines.rl.ddppo.algo import DDPPOTrainer
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.rl.ppo.encoder_dict import (
    get_vision_encoder_inputs
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)

@baseline_registry.register_trainer(name="belief-ddppo")
class BeliefDDPPOTrainer(DDPPOTrainer):
    r"""Fit DDPPO for belief work (re-overwrite DDPPO as needed)
    """
    supported_tasks = ["Nav-v0"]

    def get_ppo_class(self):
        return DDPPO

    def _setup_actor_critic_agent(self, *args, **kwargs):
        # Skip over DDPPO details
        return PPOTrainer._setup_actor_critic_agent(self, *args, **kwargs)

    def train(self, ckpt_path="", ckpt=-1, start_updates=0) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        self.local_rank, tcp_store = init_distrib_slurm(
            self.config.RL.DDPPO.distrib_backend
        )
        add_signal_handlers()

        # Stores the number of workers that have finished their rollout
        num_rollouts_done_store = distrib.PrefixStore(
            "rollout_tracker", tcp_store
        )
        num_rollouts_done_store.set("num_done", "0")

        self.world_rank = distrib.get_rank()
        self.world_size = distrib.get_world_size()

        random.seed(self.config.TASK_CONFIG.SEED + self.world_rank)
        np.random.seed(self.config.TASK_CONFIG.SEED + self.world_rank)

        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_ID = self.local_rank
        self.config.freeze()

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
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
                observation_space=self.obs_space, policy_encoders=policy_encoders_map)

        self._setup_actor_critic_agent(
            ppo_cfg, task_cfg, aux_cfg,
            init_aux_tasks,
            aux_encoders=aux_encoder_insts,
            policy_encoders=policy_encoders_map
        )
        self.agent.init_distributed(find_unused_params=True)

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

        if self.world_rank == 0:
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
        prev_time = 0

        if ckpt != -1:
            logger.info(
                f"Resuming runs at checkpoint {ckpt}. Timing statistics are not tracked properly."
            )
            assert ppo_cfg.use_linear_lr_decay is False and ppo_cfg.use_linear_clip_decay is False, "Resuming with decay not supported"
            # This is the checkpoint we start saving at
            count_checkpoints = ckpt + 1
            self.count_steps = start_updates * ppo_cfg.num_steps * self.config.NUM_PROCESSES
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
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

        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            self.agent.optimizer.load_state_dict(
                interrupted_state["optim_state"]
            )
            lr_scheduler.load_state_dict(interrupted_state["lr_sched_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            env_time = requeue_stats["env_time"]
            pth_time = requeue_stats["pth_time"]
            self.count_steps = requeue_stats["count_steps"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            start_updates = requeue_stats["start_update"]
            prev_time = requeue_stats["prev_time"]

        with (
            TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )
            if self.world_rank == 0
            else contextlib.suppress()
        ) as writer:

            for update in range(start_updates, self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                if EXIT.is_set():
                    self.envs.close()

                    if REQUEUE.is_set() and self.world_rank == 0:
                        requeue_stats = dict(
                            env_time=env_time,
                            pth_time=pth_time,
                            count_steps=self.count_steps,
                            count_checkpoints=count_checkpoints,
                            start_update=update,
                            prev_time=(time.time() - t_start) + prev_time,
                        )
                        save_interrupted_state(
                            dict(
                                state_dict=self.agent.state_dict(),
                                optim_state=self.agent.optimizer.state_dict(),
                                lr_sched_state=lr_scheduler.state_dict(),
                                config=self.config,
                                requeue_stats=requeue_stats,
                            )
                        )

                    requeue_job()
                    return

                count_steps_delta = 0
                self.agent.eval()
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
                    count_steps_delta += delta_steps

                    # This is where the preemption of workers happens.  If a
                    # worker detects it will be a straggler, it preempts itself!
                    if (
                        step
                        >= ppo_cfg.num_steps * self.SHORT_ROLLOUT_THRESHOLD
                    ) and int(num_rollouts_done_store.get("num_done")) > (
                        self.config.RL.DDPPO.sync_frac * self.world_size
                    ):
                        break

                num_rollouts_done_store.add("num_done", 1)
                self.agent.train()

                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                    aux_task_losses,
                    aux_dist_entropy,
                    aux_weights,
                    inv_curiosity,
                    fwd_curiosity,
                ) = self._update_agent(ppo_cfg, rollouts)

                pth_time += delta_pth_time

                stats_ordering = list(sorted(running_episode_stats.keys()))

                reward_ordering = [k for k in stats_ordering if k == "reward" or "env_reward" in k]
                not_reward_ordering = [k for k in stats_ordering if not(k == "reward" or "env_reward" in k)]

                # Need to pull out reward because it may not be a scalar
                reward_stats = torch.stack(
                    [running_episode_stats[k] for k in reward_ordering], 0
                ).to(self.device)
                distrib.all_reduce(reward_stats)
                not_reward_stats = torch.stack(
                    [running_episode_stats[k] for k in not_reward_ordering], 0
                ).to(self.device)
                distrib.all_reduce(not_reward_stats)

                for i, k in enumerate(reward_ordering):
                    window_episode_stats[k].append(reward_stats[i].clone())
                for i, k in enumerate(not_reward_ordering):
                    window_episode_stats[k].append(not_reward_stats[i].clone())

                stats = torch.tensor(
                    [aux_dist_entropy, inv_curiosity, fwd_curiosity] +
                    aux_task_losses + [count_steps_delta],
                    device=self.device,
                )
                distrib.all_reduce(stats)
                per_head_stats = torch.stack([dist_entropy, value_loss, action_loss], dim=0).to(self.device)
                distrib.all_reduce(per_head_stats)
                self.count_steps += stats[-1].item()
                elapsed_steps += stats[-1].item()

                if aux_weights is not None and len(aux_weights) > 0:
                    aux_weights = torch.tensor(aux_weights, device=self.device)
                    distrib.all_reduce(aux_weights)
                    aux_weights = aux_weights / self.world_size

                if self.world_rank == 0:
                    num_rollouts_done_store.set("num_done", "0")

                    avg_stats = [
                        stats[i].item() / self.world_size for i in range(len(stats) - 1)
                    ]
                    aux_dist_entropy, inv_curiosity, fwd_curiosity = avg_stats[:3]
                    aux_task_losses = avg_stats[3:]

                    avg_entropy = per_head_stats[0] / self.world_size
                    losses = per_head_stats[1:]

                    deltas = self._make_deltas(window_episode_stats)

                    self.report_train_metrics(writer, {
                        "aux_entropy": aux_dist_entropy,
                        "inv_curiosity_loss": inv_curiosity,
                        "fwd_curiosity_loss": fwd_curiosity,
                    },
                        deltas, avg_entropy, losses, aux_task_losses, self.count_steps, elapsed_steps, update,
                        env_time, pth_time, t_start, window_episode_stats,
                        aux_weights, aux_task_strings)

                    # checkpoint model
                    if update % self.config.CHECKPOINT_INTERVAL == 0:
                        self.save_checkpoint(
                            f"{self.checkpoint_prefix}.{count_checkpoints}.pth",
                            dict(step=self.count_steps)
                        )
                        count_checkpoints += 1

        self.envs.close()
