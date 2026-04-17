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


def _make_runner(num_steps_per_env: int = 2) -> ReppoRunner:
    return ReppoRunner(
        DummyEnv(),
        {
            "num_steps_per_env": num_steps_per_env,
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
                "num_action_samples": 3,
            },
        },
        log_dir=None,
        device="cpu",
    )


def test_compute_returns_uses_final_observation_and_monte_carlo_next_values() -> None:
    runner = _make_runner(num_steps_per_env=1)
    obs = runner.env.get_observations()

    def fake_sample_actions_from_normalized(self, normalized_obs: torch.Tensor):
        return (
            torch.full((1, 2), 0.25),
            torch.tensor([0.5]),
            torch.tensor([-0.5]),
            torch.full((1, 2), 0.1),
            torch.tensor(2.0),
            torch.tensor(3.0),
        )

    seen_next_obs: list[torch.Tensor] = []

    class FakeDist:
        def __init__(self, obs: torch.Tensor) -> None:
            self.obs = obs
            self._sample_cursor = 0

        def sample(self, sample_shape: tuple[int, ...] = torch.Size()) -> torch.Tensor:
            seen_next_obs.append(self.obs.clone())
            if sample_shape:
                chunk_size = sample_shape[0]
                base = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
                values = base[self._sample_cursor : self._sample_cursor + chunk_size]
                self._sample_cursor += chunk_size
                return values.view(chunk_size, 1, 1, 1).expand(chunk_size, 1, 1, 2)
            return torch.tensor([[[0.4, 0.4]]], dtype=torch.float32)

        def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
            return torch.full(actions.shape, 0.25, dtype=torch.float32)

    def fake_build_distribution_from_normalized(self, normalized_obs: torch.Tensor):
        return FakeDist(normalized_obs)

    def fake_forward_normalized(self, obs: torch.Tensor, action: torch.Tensor):
        if obs.dim() == 4:
            values = action[..., :1] * 10.0
            embeddings = torch.ones(obs.shape[0], obs.shape[1], obs.shape[2], 8, dtype=torch.float32) * 9.0
            return values, torch.zeros(*values.shape, self.num_atoms), torch.ones_like(embeddings), embeddings
        return (
            torch.tensor([[7.0]]),
            torch.zeros(1, self.num_atoms),
            torch.ones(1, 8),
            torch.ones(1, 8) * 9.0,
        )

    runner.policy.sample_actions_from_normalized = types.MethodType(fake_sample_actions_from_normalized, runner.policy)
    runner.policy.build_distribution_from_normalized = types.MethodType(
        fake_build_distribution_from_normalized, runner.policy
    )
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

    runner.compute_returns(next_obs)

    assert torch.allclose(seen_next_obs[0], final_obs["policy"].unsqueeze(0))
    assert torch.allclose(runner.rollout_data["rewards"], torch.tensor([[[4.99505]]]), atol=1e-6)
    assert torch.allclose(runner.rollout_data["next_values"], torch.tensor([[[2.0]]]), atol=1e-6)
    assert torch.allclose(runner.rollout_data["next_embeddings"], torch.ones(1, 1, 8) * 9.0)


def test_compute_returns_uses_shifted_next_actions_for_non_truncated_steps() -> None:
    runner = _make_runner()
    obs = runner.env.get_observations()

    rollout_actions = [
        torch.tensor([[0.11, 0.12]], dtype=torch.float32),
        torch.tensor([[0.21, 0.22]], dtype=torch.float32),
    ]

    def fake_sample_actions_from_normalized(self, normalized_obs: torch.Tensor):
        action = rollout_actions.pop(0)
        return (
            action,
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            action,
            torch.tensor(1.0),
            torch.tensor(1.0),
        )

    recorded_log_prob_actions: list[torch.Tensor] = []

    class FakeDist:
        def sample(self, sample_shape: tuple[int, ...] = torch.Size()) -> torch.Tensor:
            if sample_shape:
                return torch.zeros(*sample_shape, 2, 1, 2, dtype=torch.float32)
            return torch.tensor(
                [
                    [[0.31, 0.32]],
                    [[0.41, 0.42]],
                ],
                dtype=torch.float32,
            )

        def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
            recorded_log_prob_actions.append(actions.clone())
            return torch.zeros(actions.shape[:-1], dtype=torch.float32)

    def fake_build_distribution_from_normalized(self, normalized_obs: torch.Tensor):
        return FakeDist()

    def fake_forward_normalized(self, obs: torch.Tensor, action: torch.Tensor):
        value_shape = action.shape[:-1]
        values = torch.ones(value_shape, dtype=torch.float32)
        embeddings = torch.ones(*value_shape, 8, dtype=torch.float32)
        return values, torch.zeros(*value_shape, self.num_atoms), torch.ones_like(embeddings), embeddings

    runner.policy.sample_actions_from_normalized = types.MethodType(fake_sample_actions_from_normalized, runner.policy)
    runner.policy.build_distribution_from_normalized = types.MethodType(
        fake_build_distribution_from_normalized, runner.policy
    )
    runner.critic.forward_normalized = types.MethodType(fake_forward_normalized, runner.critic)

    first_obs = obs
    second_obs = TensorDict(
        {
            "policy": torch.tensor([[5.0, 6.0]], dtype=torch.float32),
            "critic": torch.tensor([[7.0, 8.0]], dtype=torch.float32),
        },
        batch_size=[1],
    )
    third_obs = TensorDict(
        {
            "policy": torch.tensor([[9.0, 10.0]], dtype=torch.float32),
            "critic": torch.tensor([[11.0, 12.0]], dtype=torch.float32),
        },
        batch_size=[1],
    )

    runner.act(first_obs)
    runner.process_env_step(second_obs, rewards=torch.tensor([1.0]), dones=torch.tensor([0.0]), extras={})
    runner.act(second_obs)
    runner.process_env_step(third_obs, rewards=torch.tensor([1.0]), dones=torch.tensor([0.0]), extras={})

    runner.compute_returns(third_obs)

    true_next_actions = recorded_log_prob_actions[0]
    assert torch.allclose(true_next_actions[0], torch.tensor([[0.21, 0.22]]), atol=1e-6)
    assert torch.allclose(true_next_actions[1], torch.tensor([[0.41, 0.42]]), atol=1e-6)


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


def test_broadcast_parameters_syncs_policy_old() -> None:
    runner = _make_runner()

    with torch.no_grad():
        for param in runner.policy.parameters():
            param.fill_(1.0)
        for param in runner.critic.parameters():
            param.fill_(2.0)
        for param in runner.policy_old.parameters():
            param.zero_()

    broadcast_payload: list[object] = []

    def fake_broadcast_object_list(obj_list: list[object], src: int) -> None:
        broadcast_payload.extend(obj_list)

    with mock.patch("torch.distributed.broadcast_object_list", side_effect=fake_broadcast_object_list):
        runner.broadcast_parameters()

    assert len(broadcast_payload) == 2
    policy_old_state = runner.policy_old.state_dict()
    policy_state = runner.policy.state_dict()
    for key, value in policy_state.items():
        assert torch.allclose(policy_old_state[key], value)
