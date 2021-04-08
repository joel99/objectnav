import abc
import math
from typing import Dict, Tuple, List
import copy
import torch
import torch.nn as nn

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import (
    CriticHead, CategoricalNet
)

from habitat_baselines.rl.ppo.belief_policy import (
    BeliefPolicy, AttentiveBeliefPolicy, AttentiveBeliefCore
)

class StackedModules(nn.Module):
    # For making multiple policy heads
    def __init__(self, base_module, copies=1, stack_dim=-2):
        super().__init__()
        self.stack_dim = stack_dim
        self.stack = nn.ModuleList([
            copy.deepcopy(base_module) for _ in range(copies)
        ])

    def forward(self, *args, **kwargs):
        results = [module(*args, **kwargs) for module in self.stack]
        return torch.stack(results, dim=self.stack_dim)

class MultiPolicyMixin(BeliefPolicy):
    def init_multipolicy(self):
        # Called after init
        self.stack_dim = -2
        # Overwrite
        self.action_distribution = StackedModules(
            CategoricalNet(self.net.output_size, self.dim_actions),
            copies=self.num_policy_heads,
            stack_dim=self.stack_dim
        )
        self.critic = StackedModules(
            CriticHead(self.net.output_size),
            copies=self.num_policy_heads,
            stack_dim=self.stack_dim
        )

    def _run_multipolicy_act(self, features, deterministic=False, behavioral_index=0):
        r"""
            value: n x k x 1
        """
        # Unfortunately, since we're accessing internal attributes in single policy already
        # We'll do so again here

        logits = self.action_distribution(features) # b x k x A
        value = self.critic(features)

        dist_results = [
            module.dist.act(
                logits[...,i, :], sample=not deterministic # Technically this should be according to stack dim
            ) for i, module in enumerate(self.action_distribution.stack)
        ]

        stacked_dist = {
            k: self._stack_dict(dist_results, k) for k in dist_results[0]
        }
        return value, stacked_dist["actions"][..., behavioral_index, :], stacked_dist, logits

    def _run_multipolicy_evaluate_actions(self, features, action):
        r"""
            returns: (guesses)
                value: b x k x h=1
                action_log_probs: b x k x A
                entropy: k
        """
        # ! It's bad if we switch in the middle of a rollout but it should just be a speed bump
        logits = self.action_distribution(features) # b x A
        value = self.critic(features)

        dist_results = [
            module.dist.evaluate_actions(
                logits[..., i, :], action # Technically this should be according to stack dim
            ) for i, module in enumerate(self.action_distribution.stack)
        ]
        stacked_dist = {
            k: self._stack_dict(dist_results, k) for k in dist_results[0] if k is not "entropy"
        }
        stacked_dist["entropy"] = torch.stack([dist["entropy"].mean() for dist in dist_results])

        return value, stacked_dist["action_log_probs"], stacked_dist["entropy"]

    def _stack_dict(self, arr_of_dicts, key):
        extracted = [d[key] for d in arr_of_dicts]
        return torch.stack(extracted, dim=self.stack_dim)

@baseline_registry.register_policy
class AttentiveBeliefMultiPolicy(MultiPolicyMixin, AttentiveBeliefPolicy):
    def __init__(self, *args, num_policy_heads=1, net=AttentiveBeliefCore, **kwargs):
        super().__init__(*args, net=net, **kwargs)
        self.num_policy_heads = num_policy_heads
        self.init_multipolicy()
        # Although mixin order may naturally work, this removes ambiguity
        self.run_policy_act = self._run_multipolicy_act
        self.run_policy_evaluate_actions = self._run_multipolicy_evaluate_actions