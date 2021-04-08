#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import math
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn

from habitat_baselines.common.utils import Flatten
import habitat_baselines.rl.models.resnet as resnet
from habitat_baselines.rl.ppo.policy import Policy, Net, GOAL_EMBEDDING_SIZE, ObservationSequential

"""
This module was experimental and used to support multiple visual streams
(e.g. separate resnets processing RGBD and Semantics).
We found little difference when splitting, but kept the module.
"""

_key_to_sensor = {
    "rgbd": ["rgb", "depth"],
    "rgbdsem": ["rgb", "depth", "semantic"],
    "none": []
}
def key_to_sensor(k):
    if k in _key_to_sensor:
        return _key_to_sensor[k]
    return [k]

def get_vision_encoder_inputs(ppo_cfg):
        r"""
            Different downstream modules will query for a certain input modality.
            Here, we map requested modalities to the inputs of not yet instantiated CNN.
        """
        policy_encoders = {}
        ENCODERS = ppo_cfg.POLICY.BELIEFS.ENCODERS
        assert len(ENCODERS) == 1 or (len(ENCODERS) == ppo_cfg.POLICY.BELIEFS.NUM_BELIEFS and "all" not in ENCODERS)
        default_sensors = key_to_sensor(ENCODERS[0])

        policy_encoders["all"] = default_sensors

        # For each visual encoder (keyed by modality) specify the sensors used
        # If a module requestss a subset of modalities (e.g. only rgb), we will give them the superset (e.g. rgbd) that is used

        if "rgb" in default_sensors:
            policy_encoders["rgb"] = default_sensors # superset
        if "depth" in default_sensors:
            policy_encoders["depth"] = default_sensors # superset
            # semantics, edge cases, aren't really thorough.

        if len(ENCODERS) == 1 and ppo_cfg.POLICY.USE_SEMANTICS:
            default_sensors.append("semantic")
            policy_encoders["semantic"] = default_sensors
        for encoder in ENCODERS:
            if encoder not in policy_encoders:
                policy_encoders[encoder] = key_to_sensor(encoder)
        return policy_encoders

class BlindDummyResnet(nn.Module):
    r"""
        Rather than have irregular visions we can't stack, just feed zero as the blind vision embedding
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
    def forward(self, observations, **kwargs):
        return torch.zeros(observations["depth"].size(0), self.hidden_size, device=observations["depth"].device) # Assuming depth

class ResnetVisionEncoderSet(nn.Module):
    r"""
        Holds onto a number of encoders, each of which can be associated with more than one label.
        Used to make sure everyone gets the right sensory information without dup-ing forward passes.
        JIT-able
    """
    def __init__(self,
        encoder_labels,
        observation_space,
        resnet_baseplanes = 32,
        backbone = "resnet18",
        hidden_size = 512,
        mock_semantics: bool = False,
        **kwargs,
    ):
        r"""
            encoder_labels: requirements dict.
                key: sensor requirement
                value: inputs to corresponding encoder (a hash for the encoder)
            **kwargs forward to resnet construction
        """
        super().__init__()
        sensor_to_encoder = {k: sorted(v) for k, v in encoder_labels.items()}
        self.encoder_labels = {k: str(v) for k, v in sensor_to_encoder.items()}
        encoders = {}
        for modalities in sensor_to_encoder.values():
            if str(modalities) in encoders:
                continue
            if len(modalities) == 0:
                encoders[str(modalities)] = BlindDummyResnet(hidden_size)
                continue

            # re: mock objectnav: Semantic space is not read so we don't have to modify space
            visual_resnet = resnet.ResNetEncoder(
                observation_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                use_if_available=modalities,
                mock_semantics=mock_semantics,
                **kwargs
                # While we ideally wouldn't record this on resnet, I don't think there's harm
                # And it's more convenient than passing arg through nn.Sequential (which is the top-level module we're using)
            )

            visual_encoder = ObservationSequential(
                visual_resnet,
                Flatten(),
                nn.Linear(
                    int(np.prod(visual_resnet.output_shape)), hidden_size # int cast for jit
                ),
                nn.ReLU(True),
            )

            encoders[str(modalities)] = visual_encoder
        self.encoders = nn.ModuleDict(encoders)

    def has_modality(self, modality):
        return modality in self.encoder_labels

    def forward(self,
        observations: Dict[str, torch.Tensor],
        # other_embeddings: Optional[List[torch.Tensor]] = None
    ):
        r"""
            Forward each encoder and assign encoder per sensor requirements.
            observations: dictionary of raw sensor inputs
            # other_embeddings: list of other embeddings to cat onto the vision embedding
        """
        embeddings = {}
        for label, encoder in self.encoders.items():
            embeddings[label] = encoder(observations) # b x h
            # if other_embeddings is not None:
            #     # Written out for JIT
            #     all_embeddings = [embeddings[label]]
            #     for other in other_embeddings:
            #         all_embeddings.append(other)
            #     embeddings[label] = torch.cat(all_embeddings, dim=-1)
                # del all_embeddings
                # embeddings[label] = torch.cat([embeddings[label], *other_embeddings], dim=-1)
        sensor_to_encoder = {}
        for k, v in self.encoder_labels.items():
            sensor_to_encoder[k] = embeddings[v]
        return sensor_to_encoder
