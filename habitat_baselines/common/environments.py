#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""
import math
import numpy as np
from typing import Optional, Type

import habitat
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.tasks.nav.nav import (
    TopDownMap
)
from habitat.tasks.nav.object_nav_task import (
    GoalObjectVisible
)

def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="NavRLEnv")
class NavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE

        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="NavExplorationRLEnv")
class NavExplorationRLEnv(NavRLEnv):
    r""" Wrapper around nav that adds an exploration bonus """
    def __init__(self, config, dataset=None): # add coverage to the metrics
        super().__init__(config, dataset)
        # self.step_count = 0
        self.step_penalty = 1

    def step(self, *args, **kwargs):
        # self.step_count += 1
        # if self.step_count > 100:
        self.step_penalty *= self._rl_config.COVERAGE_ATTENUATION
        return super().step(*args, **kwargs)

    def reset(self):
        # self.step_count = 0
        self.step_penalty = 1
        return super().reset()

    def get_reward_range(self):
        old_low, old_hi = super().get_reward_range()
        return old_low, old_hi + self._rl_config.COVERAGE_REWARD

    def get_reward(self, observations):
        visit = self.habitat_env.get_metrics()["coverage"]["visit_count"]
        if self._rl_config.COVERAGE_VISIT_EXP == 0 and visit > 1:
            exploration_reward = 0
        else:
            exploration_reward = self.step_penalty * self._rl_config.COVERAGE_REWARD / (visit ** self._rl_config.COVERAGE_VISIT_EXP)
        # if self._rl_config.COVERAGE_PRIZE_PENALTY:
        # Penalize by time the final prize - this should nullify any additional coverage
        reward = super().get_reward(observations) + exploration_reward
        return reward

@baseline_registry.register_env(name="ExploreThenNavRLEnv")
class ExploreThenNavRLEnv(NavRLEnv):
    r"""
        We want to train an agent that overfits less. We provide an exploration reward and a delayed gratification success reward.
        This is to avoid weird shaping loss.
        We provide quickly attenuating coverage reward, and a gradually increasing success reward.
    """
    def __init__(self, config, dataset=None): # add coverage to the metrics
        super().__init__(config, dataset)
        self.step_penalty = 1
        self._goal_was_seen = False # Switch for turning off coverage, turning on success shaping
        self.visit_bonus = 0
        self._previous_view = 0

    def step(self, *args, **kwargs):
        self.step_penalty *= self._rl_config.COVERAGE_ATTENUATION
        return super().step(*args, **kwargs)

    def reset(self):
        self.step_penalty = 1
        self._goal_was_seen = False
        self._previous_view = 0
        # self.visit_bonus = 0
        return super().reset()

    def get_reward_range(self):
        old_low, old_hi = super().get_reward_range()
        return old_low, old_hi + self._rl_config.COVERAGE_REWARD

    def get_reward(self, observations):
        if self._goal_was_seen:
            return super().get_reward(observations)
            # return reward
            # if self._episode_success():
            #     return reward + self.visit_bonus * self._rl_config.COVERAGE_BONUS_SCALE

        # ! If there's no GoalObjectVisible measure (i.e. eval), automatically use shaping
        if GoalObjectVisible.cls_uuid not in self.habitat_env.get_metrics() or self.habitat_env.get_metrics()[GoalObjectVisible.cls_uuid] > self._rl_config.EXPLORE_GOAL_SEEN_THRESHOLD:
            self._goal_was_seen = True
            super().get_reward(observations) # ! Hack -- clear the shaping reward from exploration phase

        # distance = self._env.get_metrics()["distance_to_goal"]
        # if distance < self._rl_config.COVERAGE_FALLOFF_RADIUS or self.step_penalty < 0.01: # ~150 steps have passed
            # self.visit_bonus = math.sqrt(self.habitat_env.get_metrics()["coverage"]["reached"])

        if self._rl_config.COVERAGE_TYPE == "VISIT":
            visit = self.habitat_env.get_metrics()["coverage"]["visit_count"]
            return self.step_penalty * self._rl_config.COVERAGE_REWARD / (visit ** self._rl_config.COVERAGE_VISIT_EXP)
            # return self.step_penalty * self._rl_config.COVERAGE_REWARD / (visit ** self._rl_config.COVERAGE_VISIT_EXP)
        else: # VIEW
            # print(self.habitat_env.get_metrics().keys())
            # Here we provide a bonus for the new view
            reward = 0
            map_measures = self.habitat_env.get_metrics()[TopDownMap.cls_uuid]
            # print(map_measures.keys())
            if map_measures:
                # print(map_measures["fog_of_war_mask"].size)
                # 18K / 1.8M, 40K  / 2.3M, 6K / 2M, 9K / 1M -> let's just divide by 50K
                # print(map_measures["fog_of_war_mask"].sum())
                explore_view = np.log(1 +
                    map_measures["fog_of_war_mask"].sum() / 50000.0
                    # / map_measures["fog_of_war_mask"].size
                ) / np.log(2)
                # import pdb
                # pdb.set_trace()
            else:
                explore_view = 0
            if self._previous_view > 0:
                reward += (
                    explore_view - self._previous_view
                ) * self._rl_config.COVERAGE_REWARD * self.step_penalty
            # print(reward)
            self._previous_view = explore_view
            return reward


@baseline_registry.register_env(name="ExploreSparseNavRLEnv")
class ExploreSparseNavRLEnv(NavRLEnv):
    r"""
        We want to train an agent that overfits less. We provide an exploration reward and a delayed gratification success reward.
        This is to avoid weird shaping loss.
        We provide quickly attenuating coverage reward, and a gradually increasing success reward.
    """
    def __init__(self, config, dataset=None): # add coverage to the metrics
        super().__init__(config, dataset)
        self.step_penalty = 1
        self._goal_was_seen = False # Switch for turning off coverage, turning on success shaping
        self.visit_bonus = 0
        self._previous_view = 0

    def step(self, *args, **kwargs):
        self.step_penalty *= self._rl_config.COVERAGE_ATTENUATION
        return super().step(*args, **kwargs)

    def reset(self):
        self.step_penalty = 1
        self._goal_was_seen = False
        self._previous_view = 0
        # self.visit_bonus = 0
        return super().reset()

    def get_reward_range(self):
        old_low, old_hi = super().get_reward_range()
        return old_low, old_hi + self._rl_config.COVERAGE_REWARD

    def get_reward(self, observations):

        visit = self.habitat_env.get_metrics()["coverage"]["visit_count"]
        cov_reward = self.step_penalty * self._rl_config.COVERAGE_REWARD / (visit ** self._rl_config.COVERAGE_VISIT_EXP)
        if self._episode_success():
            return self._rl_config.SUCCESS_REWARD + cov_reward
        return cov_reward

@baseline_registry.register_env(name="ExploreRLEnv")
class ExploreRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset]):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._previous_view = 0
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self.step_penalty = 1
        self._previous_view = 0
        observations = super().reset()
        return observations

    def step(self, *args, **kwargs):
        self.step_penalty *= self._rl_config.COVERAGE_ATTENUATION
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.COVERAGE_REWARD + 1.0,
        )

    def get_reward(self, observations):
        if self._rl_config.COVERAGE_TYPE == "VISIT":
            visit = self.habitat_env.get_metrics()["coverage"]["visit_count"]
            return self.step_penalty * self._rl_config.COVERAGE_REWARD / (visit ** self._rl_config.COVERAGE_VISIT_EXP)
        else: # VIEW
            reward = 0
            map_measures = self.habitat_env.get_metrics()[TopDownMap.cls_uuid]
            if map_measures:
                explore_view = np.log(1 +
                    map_measures["fog_of_war_mask"].sum() / 50000.0
                    # / map_measures["fog_of_war_mask"].size
                ) / np.log(2)
            else:
                explore_view = 0
            if self._previous_view > 0:
                reward += (
                    explore_view - self._previous_view
                ) * self._rl_config.COVERAGE_REWARD * self.step_penalty
            self._previous_view = explore_view
            return reward

    def _episode_success(self): # Note, this is not the "Success" measure that is reported
        if (
            self._env.task.is_stop_called
        ):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

@baseline_registry.register_env(name="DummyRLEnv")
class DummyRLEnv(habitat.RLEnv):
    # For when we have multiple policies + multiple rewards
    # It would also be valid to define a reward dictionary
    # But it's simply easier to wire the measures as reward instead
    # Note, you need to change env config to request reward metrics
    def __init__(self, config: Config, dataset: Optional[Dataset]):
        self._rl_config = config.RL
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SLACK_REWARD + 1.0,
        )

    def get_reward(self, observations):
        return self._rl_config.SLACK_REWARD

    def get_done(self, observations):
        return self._env.episode_over or self._env.task.is_stop_called

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
