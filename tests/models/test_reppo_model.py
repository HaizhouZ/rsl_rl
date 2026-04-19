# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.modules import ActorQ


def _make_obs() -> TensorDict:
    return TensorDict(
        {
            "policy": torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float32),
            "critic": torch.tensor([[0.5, 0.5], [2.0, -2.0]], dtype=torch.float32),
        },
        batch_size=[2],
    )


def test_actor_q_uses_state_dependent_std() -> None:
    obs = _make_obs()
    policy = ActorQ(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        num_actions=2,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(4,),
        critic_hidden_dims=(4, 4),
        state_dependent_std=True,
        distribution_type="normal",
    )

    with torch.no_grad():
        actor_layers = [module for module in policy.actor.modules() if isinstance(module, torch.nn.Linear)]
        first_linear = actor_layers[0]
        final_linear = actor_layers[-1]
        first_linear.weight.zero_()
        first_linear.bias.zero_()
        first_linear.weight[0, 0] = 1.0
        first_linear.weight[1, 1] = 1.0
        final_linear.weight.zero_()
        final_linear.bias.zero_()
        final_linear.weight[2, 0] = 1.0
        final_linear.weight[3, 1] = 1.0

    normalized_obs = policy.actor_obs_normalizer(policy.get_actor_obs(obs))
    policy._update_distribution(normalized_obs)
    std = policy.action_std

    assert std.shape == (2, 2)
    assert not torch.allclose(std[0], std[1])


def test_actor_q_can_use_global_std() -> None:
    obs = _make_obs()
    policy = ActorQ(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        num_actions=2,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(4,),
        critic_hidden_dims=(4, 4),
        state_dependent_std=False,
        distribution_type="normal",
    )

    normalized_obs = policy.actor_obs_normalizer(policy.get_actor_obs(obs))
    policy._update_distribution(normalized_obs)
    std = policy.action_std

    assert std.shape == (2, 2)
    assert torch.allclose(std[0], std[1])


def test_actor_q_inference_is_tanh_squashed() -> None:
    obs = _make_obs()
    policy = ActorQ(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        num_actions=2,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(4,),
        critic_hidden_dims=(4, 4),
        distribution_type="tanh",
        state_dependent_std=False,
    )

    with torch.no_grad():
        actor_layers = [module for module in policy.actor.modules() if isinstance(module, torch.nn.Linear)]
        actor_layers[0].weight.zero_()
        actor_layers[0].bias.zero_()
        actor_layers[1].weight.zero_()
        actor_layers[1].bias.copy_(torch.tensor([2.0, -2.0]))

    actions = policy.act_inference(obs)
    assert torch.allclose(actions[0], torch.tanh(torch.tensor([2.0, -2.0])), atol=1e-6)


def test_actor_q_returns_distributional_value_logits() -> None:
    obs = _make_obs()
    policy = ActorQ(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        num_actions=2,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(4,),
        critic_hidden_dims=(4, 4),
        num_critic_bins=51,
    )

    actions = torch.zeros(2, 2)
    values, logits = policy.evaluate(obs, actions, return_logits=True)

    assert values.shape == (2,)
    assert logits.shape == (2, 51)


def test_actor_q_hlgauss_embed_matches_num_bins() -> None:
    obs = _make_obs()
    policy = ActorQ(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        num_actions=2,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(4,),
        critic_hidden_dims=(4, 4),
        num_critic_bins=21,
    )

    embedded = policy.hlgauss_embed(torch.tensor([0.0, 1.0], dtype=torch.float32))

    assert embedded.shape == (2, 21)
    assert torch.allclose(embedded.sum(dim=-1), torch.ones(2), atol=1e-5)
