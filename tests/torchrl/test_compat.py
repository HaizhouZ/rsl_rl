# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for TorchRL compatibility wrapper."""

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.torchrl import TorchRLVecEnvWrapper, to_torchrl_action_tensordict


class DummyEnv(VecEnv):
    """Small VecEnv implementation for wrapper tests."""

    def __init__(self) -> None:
        self.num_envs = 3
        self.num_actions = 2
        self.max_episode_length = 10
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {}

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {
                "policy": torch.ones(self.num_envs, 4),
                "critic": 2 * torch.ones(self.num_envs, 4),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs = self.get_observations()
        rewards = actions.sum(dim=-1)
        dones = torch.tensor([0.0, 1.0, 1.0])
        extras = {"time_outs": torch.tensor([0.0, 0.0, 1.0])}
        return obs, rewards, dones, extras


def test_reset_returns_torchrl_style_tensordict() -> None:
    env = TorchRLVecEnvWrapper(DummyEnv(), observation_keys=["policy", "critic"])
    td = env.reset()

    assert "observation" in td.keys(include_nested=False)
    assert "observations" in td.keys(include_nested=False)
    assert td["observation"].shape == (3, 8)
    assert td["done"].dtype == torch.bool


def test_step_builds_next_payload_with_done_split() -> None:
    env = TorchRLVecEnvWrapper(DummyEnv())
    actions = torch.ones(3, 2)

    step_td = env.step(actions)
    next_td = step_td["next"]

    assert next_td["reward"].shape == (3, 1)
    assert torch.equal(next_td["done"].squeeze(-1), torch.tensor([False, True, True]))
    assert torch.equal(next_td["terminated"].squeeze(-1), torch.tensor([False, True, False]))
    assert torch.equal(next_td["truncated"].squeeze(-1), torch.tensor([False, False, True]))


def test_step_accepts_action_tensordict() -> None:
    env = TorchRLVecEnvWrapper(DummyEnv())
    action_td = to_torchrl_action_tensordict(torch.ones(3, 2))
    step_td = env.step(action_td)

    assert "next" in step_td.keys(include_nested=False)
