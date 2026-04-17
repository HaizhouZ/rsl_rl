# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import types
from unittest import mock

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.runners import ReppoRunner


class DummyEnv(VecEnv):
    def __init__(self) -> None:
        self.num_envs = 1
        self.num_actions = 2
        self.max_episode_length = 8
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {}
        self.unwrapped = types.SimpleNamespace(common_step_counter=0)

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {
                "policy": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                "critic": torch.tensor([[3.0, 4.0]], dtype=torch.float32),
            },
            batch_size=[1],
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        raise NotImplementedError


def _make_runner() -> ReppoRunner:
    return ReppoRunner(
        DummyEnv(),
        {
            "num_steps_per_env": 2,
            "save_interval": 100,
            "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
            "env": {"partial_reset": True, "has_final_obs": True},
            "policy": {
                "class_name": "ReppoPolicy",
                "critic_class_name": "ReppoCritic",
                "actor_hidden_dims": [8],
                "critic_hidden_dims": [8],
                "actor_obs_normalization": False,
                "critic_obs_normalization": False,
            },
            "algorithm": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "num_mini_batches": 1,
                "num_learning_epochs": 1,
            },
        },
        log_dir=None,
        device="cpu",
    )


def test_process_env_step_uses_final_observation_and_entropy_shaped_reward() -> None:
    runner = _make_runner()
    obs = runner.env.get_observations()

    seen_actor_obs: list[torch.Tensor] = []

    def fake_sample_actions_from_normalized(self, normalized_obs: torch.Tensor):
        seen_actor_obs.append(normalized_obs.clone())
        return (
            torch.full((1, 2), 0.25),
            torch.tensor([0.5]),
            torch.tensor([-0.5]),
            torch.full((1, 2), 0.1),
            torch.tensor(2.0),
            torch.tensor(3.0),
        )

    def fake_forward_normalized(self, obs: torch.Tensor, action: torch.Tensor):
        return (
            torch.tensor([[7.0]]),
            torch.zeros(1, self.num_atoms),
            torch.ones(1, 8),
            torch.ones(1, 8) * 9.0,
        )

    runner.policy.sample_actions_from_normalized = types.MethodType(fake_sample_actions_from_normalized, runner.policy)
    runner.critic.forward_normalized = types.MethodType(fake_forward_normalized, runner.critic)

    runner.act(obs)

    next_obs = TensorDict(
        {
            "policy": torch.tensor([[10.0, 10.0]], dtype=torch.float32),
            "critic": torch.tensor([[20.0, 20.0]], dtype=torch.float32),
        },
        batch_size=[1],
    )
    final_obs = TensorDict(
        {
            "policy": torch.tensor([[30.0, 30.0]], dtype=torch.float32),
            "critic": torch.tensor([[40.0, 40.0]], dtype=torch.float32),
        },
        batch_size=[1],
    )

    runner.process_env_step(
        next_obs,
        rewards=torch.tensor([5.0]),
        dones=torch.tensor([0.0]),
        extras={
            "time_outs": torch.tensor([1.0]),
            "final_observation": final_obs,
        },
    )

    transition = runner.transitions[-1]
    assert torch.allclose(seen_actor_obs[0], obs["policy"])
    assert torch.allclose(seen_actor_obs[1], final_obs["policy"])
    assert torch.allclose(transition["rewards"], torch.tensor([[4.01]]), atol=1e-6)
    assert torch.allclose(transition["next_values"], torch.tensor([[[7.0]]]))


def test_reppo_runner_initializes_distributed_process_group() -> None:
    env = DummyEnv()
    cfg = {
        "num_steps_per_env": 2,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "policy": {
            "class_name": "ReppoPolicy",
            "critic_class_name": "ReppoCritic",
            "actor_hidden_dims": [8],
            "critic_hidden_dims": [8],
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
        },
        "algorithm": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "num_mini_batches": 1,
            "num_learning_epochs": 1,
        },
    }

    with (
        mock.patch.dict(os.environ, {"WORLD_SIZE": "2", "LOCAL_RANK": "1", "RANK": "1"}, clear=False),
        mock.patch("torch.distributed.init_process_group") as init_pg,
        mock.patch("torch.cuda.set_device") as set_device,
        mock.patch("torch.nn.Module.to", autospec=True, side_effect=lambda self, *args, **kwargs: self),
        mock.patch("rsl_rl.runners.reppo_runner.Logger") as logger_cls,
    ):
        runner = ReppoRunner(env, cfg, log_dir=None, device="cuda:1")

    init_pg.assert_called_once_with(backend="nccl", rank=1, world_size=2)
    set_device.assert_called_once_with(1)
    logger_cls.assert_called_once()
    assert runner.cfg["multi_gpu"] == {"global_rank": 1, "local_rank": 1, "world_size": 2}
