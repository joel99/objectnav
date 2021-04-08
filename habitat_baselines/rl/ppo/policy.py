#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from typing import Dict

import numpy as np
from gym import spaces
import torch
from torch import nn as nn
from torch.jit import Final

from habitat.tasks.nav.nav import (
    ImageGoalSensor,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.utils import CategoricalNet, Flatten
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
import habitat_baselines.rl.models.resnet as resnet
from habitat_baselines.rl.models.resnet import ResNetEncoder
from habitat_baselines.common.running_mean_and_var import RunningMeanAndVar

from habitat_baselines.rl.models.simple_cnn import SimpleCNN

GOAL_EMBEDDING_SIZE = 32


@torch.jit.script
def _process_depth(observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "depth" in observations:
        depth_observations = observations["depth"]

        depth_observations = torch.clamp(depth_observations, 0.0, 10.0)
        depth_observations /= 10.0

        observations["depth"] = depth_observations

    return observations

class ObservationSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    r""" Sequential, but with annotation for JIT compatibility in forwarding of dict"""
    def forward(self, x: Dict[str, torch.Tensor]):
        for module in self: # copied from sequential
            x = module(x)
        return x

class Policy(nn.Module):

    # The following configurations are used in the trainer to create the appropriate rollout
    # As well as the appropriate auxiliary task wiring
    # Whether to use multiple beliefs
    IS_MULTIPLE_BELIEF = False
    # Whether to section a single belief for auxiliary tasks, keeping a single GRU core
    IS_SECTIONED = False
    # Whether the fusion module is an RNN (see RecurrentAttentivePolicy)
    IS_RECURRENT = False
    # Has JIT support
    IS_JITTABLE = False
    # Policy fuses multiple inputs
    LATE_FUSION = True

    def __init__(self, net, dim_actions, observation_space=None, config=None, **kwargs):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        actor_head_layers = getattr(config, "ACTOR_HEAD_LAYERS", 1)
        critic_head_layers = getattr(config, "CRITIC_HEAD_LAYERS", 1)

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions, layers=actor_head_layers
        )
        self.critic = CriticHead(self.net.output_size, layers=critic_head_layers)
        if "rgb" in observation_space.spaces:
            self.running_mean_and_var = RunningMeanAndVar(
                observation_space.spaces["rgb"].shape[-1]
                + (
                    observation_space.spaces["depth"].shape[-1]
                    if "depth" in observation_space.spaces
                    else 0
                ),
                initial_count=1e4,
            )
        else:
            self.running_mean_and_var = None

    def forward(self, *x):
        raise NotImplementedError

    def _preprocess_obs(self, observations):
        dtype = next(self.parameters()).dtype
        observations = {k: v.to(dtype=dtype) for k, v in observations.items()}
         # since this seems to be what running_mean_and_var is expecting
        observations = {k: v.permute(0, 3, 1, 2) if len(v.size()) == 4 else v for k, v in observations.items()}
        observations = _process_depth(observations)

        if "rgb" in observations:
            rgb = observations["rgb"].to(dtype=next(self.parameters()).dtype) / 255.0
            x = [rgb]
            if "depth" in observations:
                x.append(observations["depth"])

            x = self.running_mean_and_var(torch.cat(x, 1))
            # this preprocesses depth and rgb, but not semantics. we're still embedding that in our encoder
            observations["rgb"] = x[:, 0:3]
            if "depth" in observations:
                observations["depth"] = x[:, 3:]
        # ! Permute them back, because the rest of our code expects unpermuted
        observations = {k: v.permute(0, 2, 3, 1) if len(v.size()) == 4 else v for k, v in observations.items()}

        return observations

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        **kwargs
    ):
        observations = self._preprocess_obs(observations)
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        observations = self._preprocess_obs(observations)
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        observations = self._preprocess_obs(observations)
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, features, None, None, None


class CriticHead(nn.Module):
    HIDDEN_SIZE = 32
    def __init__(self, input_size, layers=1):
        super().__init__()
        if layers == 1:
            self.fc = nn.Linear(input_size, 1)
            nn.init.orthogonal_(self.fc.weight)
            nn.init.constant_(self.fc.bias, 0)
        else: # Only support 2 layers max
            self.fc = nn.Sequential(
                nn.Linear(input_size, self.HIDDEN_SIZE),
                nn.ReLU(),
                nn.Linear(self.HIDDEN_SIZE, 1)
            )
            nn.init.orthogonal_(self.fc[0].weight)
            nn.init.constant_(self.fc[0].bias, 0)

    def forward(self, x):
        return self.fc(x)

@baseline_registry.register_policy
class PointNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid=None,
        hidden_size=512,
        **kwargs,
    ):
        super().__init__(
            BaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
            ),
            action_space.n,
        )

class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass

class BaselineNet(Net):
    r"""Network which passes the input image through CNN and passes through RNN.
    """

    def __init__(
        self,
        observation_space,
        hidden_size,
        goal_sensor_uuid=None,
        additional_sensors=[] # low dim sensors corresponding to registered name
    ):
        # TODO OURS
        self.goal_sensor_uuid = goal_sensor_uuid
        self.additional_sensors = additional_sensors
        self._n_input_goal = 0
        if goal_sensor_uuid is not None and goal_sensor_uuid != "no_sensor":
            self.goal_sensor_uuid = goal_sensor_uuid
            self._initialize_goal_encoder(observation_space)
        # END

        self._hidden_size = hidden_size

        resnet_baseplanes = 32
        backbone="resnet18"
        visual_resnet = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
        )
        self.visual_encoder = ObservationSequential(
            visual_resnet,
            Flatten(),
            nn.Linear(
                np.prod(visual_resnet.output_shape), hidden_size
            ),
            nn.ReLU(True),
        )

        final_embedding_size = (0 if self.is_blind else self._hidden_size) + self._n_input_goal
        for sensor in additional_sensors:
            final_embedding_size += observation_space.spaces[sensor].shape[0]

        self.state_encoder = RNNStateEncoder(final_embedding_size, self._hidden_size)
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _initialize_goal_encoder(self, observation_space):
        if self.goal_sensor_uuid == ImageGoalSensor.cls_uuid:
            goal_observation_space = spaces.Dict(
                {"rgb": observation_space.spaces[ImageGoalSensor.cls_uuid]}
            )
            self.goal_visual_encoder = SimpleCNN(
                goal_observation_space, self._hidden_size
            )
            self._n_input_goal = self._hidden_size
        else:
            self._n_input_goal = observation_space.spaces[
                self.goal_sensor_uuid
            ].shape[0]
        # if (
        #     IntegratedPointGoalGPSAndCompassSensor.cls_uuid
        #     in observation_space.spaces
        # ):
        #     self._n_input_goal = observation_space.spaces[
        #         IntegratedPointGoalGPSAndCompassSensor.cls_uuid
        #     ].shape[0]
        # elif PointGoalSensor.cls_uuid in observation_space.spaces:
        #     self._n_input_goal = observation_space.spaces[
        #         PointGoalSensor.cls_uuid
        #     ].shape[0]
        # elif ImageGoalSensor.cls_uuid in observation_space.spaces:
        #     goal_observation_space = spaces.Dict(
        #         {"rgb": observation_space.spaces[ImageGoalSensor.cls_uuid]}
        #     )
        #     self.goal_visual_encoder = SimpleCNN(
        #         goal_observation_space, hidden_size
        #     )
        #     self._n_input_goal = hidden_size


    def get_target_encoding(self, observations):
        if self.goal_sensor_uuid == ImageGoalSensor.cls_uuid:
            image_goal = observations[ImageGoalSensor.cls_uuid]
            return self.goal_visual_encoder({"rgb": image_goal})
        # if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
        #     target_encoding = observations[
        #         IntegratedPointGoalGPSAndCompassSensor.cls_uuid
        #     ]

        # elif PointGoalSensor.cls_uuid in observations:
        #     target_encoding = observations[PointGoalSensor.cls_uuid]
        # elif ImageGoalSensor.cls_uuid in observations:
        #     image_goal = observations[ImageGoalSensor.cls_uuid]
        #     target_encoding = self.goal_visual_encoder({"rgb": image_goal})

        return observations[self.goal_sensor_uuid]

    def _append_additional_sensors(self, x, observations):
        for sensor in self.additional_sensors:
            x.append(observations[sensor])
        return x

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x.append(perception_embed)
        if self.goal_sensor_uuid is not None:
            x.append(self.get_target_encoding(observations))

        x = self._append_additional_sensors(x, observations)

        x = torch.cat(x, dim=-1) # t x n x -1

        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        return x, rnn_hidden_states

@baseline_registry.register_policy
class ObjectNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid=None,
        hidden_size=512,
        **kwargs,
    ):
        super().__init__(
            BaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                additional_sensors=["gps", "compass"]
            ),
            action_space.n,
        )

    def _initialize_goal_encoder(self, observation_space):
        self._n_input_goal = GOAL_EMBEDDING_SIZE
        goal_space = observation_space.spaces[
            self.goal_sensor_uuid
        ]
        self.goal_embedder = nn.Embedding(goal_space.high + 1, self._n_input_goal) # low is 0, high is given (see object nav task)

    def get_target_encoding(self, observations):
        return self.goal_embedder(observations[self.goal_sensor_uuid])
