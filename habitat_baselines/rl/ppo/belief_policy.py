#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import math
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from habitat.tasks.nav.object_nav_task import task_cat2mpcat40
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Policy, Net, GOAL_EMBEDDING_SIZE
from habitat_baselines.rl.ppo.encoder_dict import ResnetVisionEncoderSet

# -----------------------------------------------------------------------------
# Recurrent cores
# -----------------------------------------------------------------------------

class SingleBeliefCore(Net):
    r"""
        This core only wraps the recurrent part of the agent.
        It expects sensor embeddings as input (i.e. vision, or encoded goal)
    """
    def __init__(
        self,
        hidden_size=None,
        embedding_size=None,
        device=None,
        input_drop=0.0,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self._hidden_size = hidden_size
        self._embedding_size = embedding_size
        self._initialize_state_encoder()
        self.input_drop = nn.Dropout(input_drop)

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def set_streams(self, streams):
        pass

    def _initialize_state_encoder(self):
        self.state_encoder = RNNStateEncoder(self._embedding_size, self._hidden_size)

    def forward(self, obs, rnn_hidden_states, masks):
        r"""
            Roll forward the state encoder. May or may not be batched for `evaluate_actions`.
            Args:
            # ! I think the shapes are flipped here...?
                obs: observation encoding. b x h
                rnn_hidden_states: hidden states, either (t*b) or (b) x h
        """
        obs = self.input_drop(obs)
        return self.state_encoder(obs, rnn_hidden_states, masks)

class MultipleBeliefCore(SingleBeliefCore):
    r"""
        Abstract class for processing multiple belief modules.
        Responsible for returning a single fused representation.
    """
    def __init__(
        self,
        num_tasks=None,
        fusion_obs_index=0,
        num_modules=-1,
        **kwargs
    ):
        self.num_tasks = num_tasks
        self.num_modules = num_tasks if num_modules == -1 else num_modules
        self.fusion_obs_index = fusion_obs_index
        super().__init__(**kwargs)
        self._initialize_fusion_net()
        # self.cuda_streams = None

    @property
    def num_recurrent_layers(self):
        return self.state_encoders[0].num_recurrent_layers

    def _initialize_state_encoder(self):
        self.state_encoders = nn.ModuleList([
            RNNStateEncoder(self._embedding_size, self._hidden_size) for _ in range(self.num_modules)
        ])

    # def set_streams(self, streams):
    #     self.cuda_streams = streams

    def _initialize_fusion_net(self):
        r"""
            Initialize any fusion modules here.
        """
        pass

    @abc.abstractmethod
    def _fuse_beliefs(self, beliefs, obs):
        r"""
            Fusion step. Batch may or may not have time flattened.
            Args:
                beliefs: batch x module x hidden. b x k x h
                obs: b x h
        """
        pass

    def _step_rnn(self, obs, rnn_hidden_states, masks):
        r"""
            obs: b x k x h

            Step recurrent state through `obs`.
            Returns:
                beliefs - b x k x h
                rnn_hidden_states - l x n x k x h (or n x k x h?)
                masks - b x 1
        """
        embeddings = []
        all_states = []
        for i, encoder in enumerate(self.state_encoders):
            enc_embed, enc_state = encoder(obs[..., i, :], rnn_hidden_states[:, :, i], masks)
            embeddings.append(enc_embed)
            all_states.append(enc_state)

        # if self.cuda_streams is None:
        #     outputs = [encoder(obs, rnn_hidden_states[:, :, i], masks) for i, encoder in enumerate(self.state_encoders)]
        # else:
        #     outputs = [None] * self.num_tasks
        #     torch.cuda.synchronize()
        #     for i, encoder in enumerate(self.state_encoders):
        #         with torch.cuda.stream(self.cuda_streams[i]):
        #             outputs[i] = encoder(obs, rnn_hidden_states[:, :, i], masks)
        #     torch.cuda.synchronize()
        # embeddings, rnn_hidden_states = zip(*outputs) # b x h, n x h
        beliefs = torch.stack(embeddings, dim=-2) # b x k x h
        rnn_hidden_states = torch.stack(all_states, dim=-2) # n x k x h
        # rnn_hidden_states = torch.stack(rnn_hidden_states, dim=-2) # n x k x h
        return beliefs, rnn_hidden_states

    def forward(self, obs, rnn_hidden_states, masks):
        r"""
            Roll forward the recurrent core.
            Args:
                obs: (t or t*n) b x k x h
                rnn_hidden_states: l x n x k x h
        """
        # Ok, now they're different observations... what do
        obs = self.input_drop(obs)
        beliefs, rnn_hidden_states = self._step_rnn(obs, rnn_hidden_states, masks)
        contextual_embedding, weights = self._fuse_beliefs(beliefs, obs[:, self.fusion_obs_index])
        return contextual_embedding, rnn_hidden_states, beliefs, weights

class AttentiveBeliefCore(MultipleBeliefCore):
    def _initialize_fusion_net(self):
        r"""
            Dot-product attention conditioned on vision.
        """
        self.fusion_key = nn.Linear(
            self._embedding_size, self._hidden_size
        )
        self.scale = math.sqrt(self._hidden_size)

    def _fuse_beliefs(self, beliefs, obs):
        key = self.fusion_key(obs) # .unsqueeze(-2)) # b x 1 x h
        scores = torch.bmm(beliefs, key.unsqueeze(-1)) / self.scale
        # scores = torch.bmm(beliefs, key.transpose(1, 2)) / self.scale
        weights = F.softmax(scores, dim=1).squeeze(-1) # n x k (logits) x 1 -> (txn) x k
        # n x 1 x k x n x k x h
        contextual_embedding = torch.bmm(weights.unsqueeze(1), beliefs).squeeze(1) # txn x h
        return contextual_embedding, weights

class RecurrentAttentiveBeliefCore(MultipleBeliefCore):
    def _initialize_fusion_net(self):
        # Output txn x 1 x h at each timestep
        self.attention_rnn = RNNStateEncoder(self._embedding_size, self._hidden_size)
        self.scale = math.sqrt(self._hidden_size)

    # belief: (txn) x k x h - I can't separate out the batch erk
    def _fuse_beliefs(self, beliefs, obs, attention_hidden_states, masks, *args):
        # obs: b x h
        # belief: b x k x h
        # attention_hidden_states: 1 (l) x b x h
        key, attention_hidden_states = self.attention_rnn(obs, attention_hidden_states, masks) # key: b x h
        scores = torch.bmm(beliefs, key.unsqueeze(-1)) / self.scale # b x h x 1
        weights = F.softmax(scores, dim=1).squeeze(-1) # n x k (logits) x 1 -> (txn) x k
        # n x 1 x k x n x k x h
        contextual_embedding = torch.bmm(weights.unsqueeze(1), beliefs).squeeze(1) # txn x h

        return contextual_embedding, weights, attention_hidden_states

    def forward(self, obs, rnn_hidden_states, masks):
        obs = self.input_drop(obs) # ok, what's determining the number of obs dims?
        # attention_hidden_states = rnn_hidden_states[:, :, -1]
        beliefs, rnn_hidden_states[:, :, :-1] = self._step_rnn(obs, rnn_hidden_states[:, :, :-1], masks)
        contextual_embedding, weights, rnn_hidden_states[:, :, -1] = self._fuse_beliefs(
            beliefs,
            obs[:, self.fusion_obs_index],
            rnn_hidden_states[:, :, -1],
            masks
        )
        # rnn_hidden_states = torch.stack((*rnn_hidden_states, attention_hidden_states.unsqueeze(-2)), dim=-2) # (layers) x n x k x h
        return contextual_embedding, rnn_hidden_states, beliefs, weights
# -----------------------------------------------------------------------------
# Policies
# -----------------------------------------------------------------------------

# ! Not everything is JIT-compatible yet. We try to JIT the major components where we can.

@baseline_registry.register_policy
class BeliefPolicy(Policy):
    r""" Encodes sensor readings and forwards to a recurrent net. """
    ACTION_EMBEDDING_SIZE = 4
    IS_JITTABLE = True

    def __init__(
        self,
        observation_space=None,
        action_space=None,
        hidden_size=None,
        net=SingleBeliefCore,
        num_tasks=0,
        goal_sensor_uuid=None,
        additional_sensors=[],
        embed_goal=False,
        config=None,
        policy_encoders={"all": ["rgb", "depth"]},
        device=None,
        mock_objectnav=False,
        **kwargs # * forwarded to net
    ):
        assert issubclass(net, SingleBeliefCore), "Belief policy must use belief net"

        # * Calculate embedding size (inputs to recurrent net)
        n_input_goal = 0
        if goal_sensor_uuid is not None and goal_sensor_uuid != "no_sensor":
            n_input_goal = observation_space.spaces[
                goal_sensor_uuid
            ].shape[0]
            if embed_goal:
                n_input_goal = GOAL_EMBEDDING_SIZE
        self.mock_objectnav = mock_objectnav
        if mock_objectnav:
            assert n_input_goal == 0, "goal shouldn't have been set yet"
            n_input_goal = GOAL_EMBEDDING_SIZE
        embedding_size = hidden_size + n_input_goal
        for sensor in additional_sensors:
            embedding_size += observation_space.spaces[sensor].shape[0]
        if config.embed_actions:
            embedding_size += self.ACTION_EMBEDDING_SIZE
        if config.embed_sge:
            embedding_size += 1
        self.embedding_size = embedding_size

        # * Obs - belief mapping
        self.num_tasks = num_tasks
        self.obs_belief_map = config.BELIEFS.ENCODERS # an array of sensor requirements for each module (only used in multiplebelief)
        if len(self.obs_belief_map) < self.num_tasks: # we assume "all"
            self.obs_belief_map = ["all"] * self.num_tasks
        fusion_obs_index = 0
        if config.BELIEFS.OBS_KEY in self.obs_belief_map:
            fusion_obs_index = self.obs_belief_map.index(config.BELIEFS.OBS_KEY)
             # though technically it just needs to be supported by policy encoders

        super().__init__(net(
                hidden_size=hidden_size,
                embedding_size=embedding_size,
                config=config, # ! Get rid of this so we can JIT.
                num_tasks=num_tasks,
                num_modules=config.BELIEFS.NUM_BELIEFS,
                fusion_obs_index=fusion_obs_index,
                input_drop=config.input_drop,
                device=device,
                **kwargs,
            ), action_space.n,
            observation_space=observation_space,
            config=config
        )

        self.num_recurrent_layers = self.net.num_recurrent_layers
        self.output_drop = nn.Dropout(config.output_drop)
        if config.use_cuda_streams:
            assert not self.IS_JITTABLE, "Compatibility not checked"
            self.cuda_streams = [torch.cuda.Stream() for i in range(num_tasks)]
            self.net.set_streams(self.cuda_streams)

        # * Init sensors
        # obs_kwargs = {}
        # if config.FULL_RESNET:
        #     obs_kwargs = { "obs_transform": None }
        self.visual_encoders = ResnetVisionEncoderSet(
            policy_encoders, observation_space,
            hidden_size=hidden_size,
            mock_semantics=self.mock_objectnav,
            # **obs_kwargs
        )
        if self.IS_JITTABLE and config.jit:
            pass # not implemented
        self.device = device

        self.additional_sensors = additional_sensors

        self.goal_sensor_uuid = goal_sensor_uuid
        self.embed_goal = embed_goal
        if goal_sensor_uuid is not None and goal_sensor_uuid != "no_sensor":
            self.goal_sensor_uuid = goal_sensor_uuid
            self._initialize_goal_encoder(observation_space)

        self.embed_actions = config.embed_actions
        if self.embed_actions:
            self._initialize_action_encoder(action_space)

        self.embed_sge = config.embed_sge
        if self.embed_sge:
            self.task_cat2mpcat40 = torch.tensor(task_cat2mpcat40, device=self.device)

    def _initialize_action_encoder(self, action_space):
        self.action_embedder = nn.Embedding(
            action_space.n + 1, self.ACTION_EMBEDDING_SIZE
        )
        if self.IS_JITTABLE:
            self.action_embedder = torch.jit.script(self.action_embedder)

    def _initialize_goal_encoder(self, observation_space):
        if not self.embed_goal or self.mock_objectnav:
            return
        goal_space = observation_space.spaces[
            self.goal_sensor_uuid
        ]
        self.goal_embedder = nn.Embedding(
            int(goal_space.high[0] - goal_space.low[0] + 1),
            GOAL_EMBEDDING_SIZE
        )
        if self.IS_JITTABLE:
            self.goal_embedder = torch.jit.script(self.goal_embedder)

    def _extract_sge(self, observations):
        # recalculating to keep this self-contained instead of depending on training infra
        if self.mock_objectnav:
            return torch.zeros(observations["semantic"].size(0), 1, device=self.device, dtype=next(self.parameters()).dtype)
        if "semantic" in observations and "objectgoal" in observations:
            obj_semantic = observations["semantic"].flatten(start_dim=1)
            idx = self.task_cat2mpcat40[
                observations["objectgoal"].long()
            ]

            goal_visible_pixels = (obj_semantic == idx).sum(dim=1) # Sum over all since we're not batched
            goal_visible_area = torch.true_divide(goal_visible_pixels, obj_semantic.size(-1))

            return goal_visible_area.unsqueeze(-1)


    def _get_observation_embeddings(self, observations, prev_actions):
        r"""
            Retrieve vision embeddings and low-d sensor embeddings.
            Kept separate to concat when needed.
            Args:
                observations: dictionary of sensory inputs
                prev_actions: N x 1
            Returns:
                vision_embeddings: b x h1
                other_embeddings: List[b x h2]
                (Should be concat-ed on last dim for use)
        """
        observations  = self._preprocess_obs(observations)
        lowd_embedding = []

        if self.goal_sensor_uuid is not None:
            if self.mock_objectnav:
                goal = torch.zeros(
                    prev_actions.size(0), # most convenient way to get batch, not ideal
                    GOAL_EMBEDDING_SIZE,
                    device=self.device,
                    dtype=next(self.parameters()).dtype
                )
            else:
                goal = observations[self.goal_sensor_uuid]
                if self.embed_goal:
                    goal = self.goal_embedder(goal.long()).squeeze(-2)
            lowd_embedding.append(goal)
        for sensor in self.additional_sensors:
            lowd_embedding.append(observations[sensor])
        if self.embed_actions:
            lowd_embedding.append(self.action_embedder(prev_actions.squeeze(-1)))
        if self.embed_sge:
            lowd_embedding.append(self._extract_sge(observations))
        return self.visual_encoders(observations), lowd_embedding

    @torch.jit.export
    def get_observation_embeddings(
        self,
        observations: Dict[str, torch.Tensor],
        prev_actions
    ):
        r"""
        Public API for curiosity
        """
        vision, other = self._get_observation_embeddings(observations, prev_actions)
        return {
            key: torch.cat([embedding, *other], dim=-1) for key, embedding in vision.items()
        }

    def get_observations_for_beliefs(self, observations, prev_actions):
        vision, other = self._get_observation_embeddings(observations, prev_actions)
        concat_obs = {
            key: torch.cat([embedding, *other], dim=-1) for key, embedding in vision.items()
        }
        return concat_obs["all"], concat_obs

    def run_policy_act(self, features, deterministic=False, **kwargs):
        # Wrapper added to deal with multipolicy
        logits = self.action_distribution(features) # b x A
        value = self.critic(features)

        dist_result = self.action_distribution.dist.act(
            logits, sample=not deterministic
        )
        return value, dist_result["actions"], dist_result, logits

    @torch.jit.export
    def act(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        return_features=False,
        return_all_activations=False,
        behavioral_index=0, # for multi-policy. Bounced in single policy
        **kwargs
    ):
        r"""
            Act by taking one step.
            Args:
                observations - dictionary of sensory input
                rnn_hidden_states - hidden states at start of step (lxnxh)
                prev_actions - previous actions (n)
                masks - for episode transitions (n x 1?)
                return_features - return encoded observations (dict: n x h)
            Returns:
                value - of input step - (n)
                action - proposed action (argmax or sampled) - (nx1)
                action_log_probs - action probability distribution - (nxA)
                rnn_hidden_states - updated hidden states of recurrent core (lxnxh)
        """
        obs, _ = self.get_observations_for_beliefs(observations, prev_actions)
        features, rnn_hidden_states = self.net(
            obs, rnn_hidden_states, masks # There's only one module, default to feeding in all
        )
        features = self.output_drop(features)

        value, actions, dist, logits = self.run_policy_act(features, deterministic=deterministic, behavioral_index=0)
        action_log_probs = dist["action_log_probs"]
        if return_all_activations:
            return (
                value, actions, action_log_probs, rnn_hidden_states,
                features, # fused state used for actions
                obs, # sensor embedding
                logits, # action logits
            )
        if return_features:
            return value, actions, action_log_probs, rnn_hidden_states, obs
        return value, actions, action_log_probs, rnn_hidden_states

    @torch.jit.export
    def get_value(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
    ):
        obs, _ = self.get_observations_for_beliefs(observations, prev_actions)
        features, *_ = self.net(
            obs, rnn_hidden_states, masks
        )
        features = self.output_drop(features)
        return self.critic(features)

    def run_policy_evaluate_actions(self, features, action, **kwargs):
        # Wrapper added to deal with multipolicy
        logits = self.action_distribution(features) # b x A
        value = self.critic(features)

        dist_result = self.action_distribution.dist.evaluate_actions(
            logits, action
        )
        return value, dist_result["action_log_probs"], dist_result["entropy"].mean()

    @torch.jit.export
    def evaluate_actions(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor, # JIT is not actually supported, this is dead code.
        torch.Tensor,
        torch.Tensor,
        List[torch.Tensor],
        torch.Tensor,
        torch.Tensor]:
        r"""
            Evaluate rollout actions.
            Args:
                observations - dictionary of sensor input
                rnn_hidden_states - l * n * k * h
                prev_actions - (t*n) x 1 (offset) - taken from previous rollout
                masks - episode transition steps
                action - actions taken ((t*n)xh)
        """
        # Concat manually so that we can return vision and other sensors separately
        # Vision is used in isolation for aux tasks (as e.g. GPS-Compass would make contrastive tasks trivial)
        vision, other = self._get_observation_embeddings(observations, prev_actions)
        obs = torch.cat([vision["all"], *other], dim=-1)

        features, rnn_hidden_states = self.net(
            obs, rnn_hidden_states, masks
        )
        features = self.output_drop(features)

        value, action_log_probs, entropy = self.run_policy_evaluate_actions(features, action)

        # Nones: individual_features, aux entropy, aux weights
        return (
            value,
            action_log_probs,
            entropy,
            rnn_hidden_states,
            features,
            None,
            None,
            None,
            self._preprocess_obs(observations), # Temp bug
            vision,
            other
            # obs_dict
        )

    def after_update(self):
        pass

class MultipleBeliefPolicy(BeliefPolicy):
    r"""Parent class for policies that use multiple recurrent modules.
        Handles auxiliary task wiring. """

    IS_MULTIPLE_BELIEF = True
    LATE_FUSION = True

    def __init__(self, net=None, config=None, **kwargs):
        assert issubclass(net, MultipleBeliefCore), "Multiple belief policy requires compatible multiple belief net"
        super().__init__(net=net, config=config, **kwargs)

        self._double_preprocess_bug = config.DOUBLE_PREPROCESS_BUG # Unfortunate but maintained so we can move forward
        # assert self.visual_encoders.has_modality(config.BELIEFS.FUSION_KEY)

    def get_observations_for_beliefs(self, observations, prev_actions):
        # Return b x k x h
        vision, other = self._get_observation_embeddings(observations, prev_actions)
        obs = {
            key: torch.cat([embedding, *other], dim=-1) for key, embedding in vision.items()
        }
        return torch.stack([obs[req] for req in self.obs_belief_map], dim=-2), obs

    @torch.jit.export
    def act(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        weights_output=None,
        return_features=False,
        return_all_activations=False,
        behavioral_index=0, # for multi-policy. Bounced in single policy
    ):
        # FIXME phase out
        if self._double_preprocess_bug:
            observations = self._preprocess_obs(observations)
        obs, obs_dict = self.get_observations_for_beliefs(observations, prev_actions) # dictionary of sensor to representations

        features, rnn_hidden_states, _, weights = self.net(
            obs, rnn_hidden_states, masks
        )
        features = self.output_drop(features)

        if weights_output is not None:
            weights_output.copy_(weights)

        value, actions, dist, logits = self.run_policy_act(features, deterministic=deterministic, behavioral_index=behavioral_index)
        action_log_probs = dist["action_log_probs"]

        if return_all_activations:
            return (
                value, actions, action_log_probs, rnn_hidden_states,
                features, # fused state used for actions
                obs, # sensor embedding
                logits, # action logits
            )
        if return_features:
            return value, actions, action_log_probs, rnn_hidden_states, obs_dict
        return value, actions, action_log_probs, rnn_hidden_states

    @torch.jit.export
    def evaluate_actions(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
            Args:
                observations - (t*n) x sensor size
                rnn_hidden_states - l=1 x n (env) x k x h
                prev_actions - (t*n) x 1 (offset) - taken from previous rollout
                masks - (t*n) x 1
                action - (t*n) x 1
            Returns:
                value
                action_log_probs
                distribution_entropy
                rnn_hidden_states - final belief hidden states l x n x k x h
                features - fused belief output (t*n) x h
                individual_features - individual belief outputs. (t*n) x k x h
                aux_dist_entropy - Note the variable is named "aux" since beliefs were developed with aux tasks in mind.
        """
        # FIXME phase out
        if self._double_preprocess_bug:
            observations = self._preprocess_obs(observations)

        vision, other = self._get_observation_embeddings(observations, prev_actions)
        obs_dict = {
            key: torch.cat([embedding, *other], dim=-1) for key, embedding in vision.items()
        }
        obs = torch.stack([obs_dict[req] for req in self.obs_belief_map], dim=-2)

        features, rnn_hidden_states, individual_features, weights = self.net(
            obs, rnn_hidden_states, masks
        )
        features = self.output_drop(features)

        value, action_log_probs, entropy = self.run_policy_evaluate_actions(features, action)

        aux_dist_entropy = None if weights is None else Categorical(weights).entropy().mean()
        weights = None if weights is None else weights.mean(dim=0) # sorry about this

        return (
            value,
            action_log_probs,
            entropy,
            rnn_hidden_states,
            features,
            individual_features,
            aux_dist_entropy,
            weights,
            observations, # preprocessed
            vision,
            other,
        )

@baseline_registry.register_policy
class AttentiveBeliefPolicy(MultipleBeliefPolicy):
    def __init__(self, *args, net=AttentiveBeliefCore, **kwargs):
        super().__init__(*args, net=net, **kwargs)

@baseline_registry.register_policy
class RecurrentAttentiveBeliefPolicy(MultipleBeliefPolicy):
    IS_RECURRENT = True
    def __init__(self, *args, net=RecurrentAttentiveBeliefCore, **kwargs):
        super().__init__(*args, net=net, **kwargs)
