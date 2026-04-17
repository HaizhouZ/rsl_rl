# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models import ReppoPolicy


def _make_obs() -> TensorDict:
    return TensorDict(
        {
            "policy": torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float32),
            "critic": torch.tensor([[0.5, 0.5], [2.0, -2.0]], dtype=torch.float32),
        },
        batch_size=[2],
    )


def test_reppo_policy_uses_state_dependent_std() -> None:
    obs = _make_obs()
    policy = ReppoPolicy(
        obs,
        {"actor": ["policy"], "critic": ["critic"]},
        num_actions=2,
        actor_obs_normalization=False,
        actor_hidden_dims=(4,),
        actor_min_std=0.1,
        use_actor_norm=False,
    )

    with torch.no_grad():
        linear_layers = [module for module in policy.actor_model.modules() if isinstance(module, torch.nn.Linear)]
        linear_layers[0].weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, -1.0],
                    [-1.0, 1.0],
                ]
            )
        )
        linear_layers[0].bias.zero_()
        linear_layers[1].weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
        linear_layers[1].bias.zero_()

    norm_obs = policy.normalize_actor_obs(obs)
    dist = policy.build_distribution_from_normalized(norm_obs)
    std = dist.base_dist.scale

    assert std.shape == (2, 2)
    assert not torch.allclose(std[0], std[1]), "REPPO actor std should depend on the observation"
