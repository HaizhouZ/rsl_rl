# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import tempfile
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
        self.cfg = {"scale_rewards_by_dt": False}
        self.unwrapped = types.SimpleNamespace(common_step_counter=0, step_dt=0.02)

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


def _make_runner(num_steps_per_env: int = 2, policy_overrides: dict | None = None) -> ReppoRunner:
    policy_cfg = {
        "class_name": "ReppoPolicy",
        "critic_class_name": "ReppoCritic",
        "actor_hidden_dims": [8],
        "critic_hidden_dims": [8],
        "actor_obs_normalization": False,
        "critic_obs_normalization": False,
    }
    if policy_overrides:
        policy_cfg.update(policy_overrides)
    return ReppoRunner(
        DummyEnv(),
        {
            "num_steps_per_env": num_steps_per_env,
            "save_interval": 100,
            "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
            "env": {"partial_reset": True, "has_final_obs": True},
            "policy": policy_cfg,
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


def test_compute_returns_uses_final_observation_for_single_sample_next_targets() -> None:
    runner = _make_runner(num_steps_per_env=1)
    obs = runner.env.get_observations()

    def fake_sample_actions_from_normalized(self, normalized_obs: torch.Tensor):
        return (
            torch.full((1, 2), 0.25),
            torch.tensor([0.5]),
            torch.tensor([-0.5]),
            torch.tensor([2.5]),
            torch.full((1, 2), 0.1),
            torch.tensor(2.0),
            torch.tensor(3.0),
        )

    seen_next_obs: list[torch.Tensor] = []

    class FakeDist:
        def __init__(self, obs: torch.Tensor) -> None:
            self.obs = obs

        def sample(self, sample_shape: tuple[int, ...] = torch.Size()) -> torch.Tensor:
            seen_next_obs.append(self.obs.clone())
            return torch.tensor([[[0.4, 0.4]]], dtype=torch.float32)

        def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
            return torch.full(actions.shape, 0.25, dtype=torch.float32)

    def fake_build_distribution_from_normalized(self, normalized_obs: torch.Tensor):
        return FakeDist(normalized_obs)

    def fake_forward_normalized(self, obs: torch.Tensor, action: torch.Tensor):
        values = action[..., :1] * 10.0
        embeddings = torch.ones(*values.shape[:-1], 8, dtype=torch.float32) * 9.0
        return values, torch.zeros(*values.shape, self.num_atoms), torch.ones_like(embeddings), embeddings

    runner.policy.sample_actions_from_normalized = types.MethodType(fake_sample_actions_from_normalized, runner.policy)
    runner.policy.build_distribution_from_normalized = types.MethodType(
        fake_build_distribution_from_normalized, runner.policy
    )
    runner.critic.forward_normalized = types.MethodType(fake_forward_normalized, runner.critic)
    with torch.no_grad():
        runner.policy.log_temp.fill_(0.0)

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
    assert torch.allclose(runner.rollout_data["rewards"], torch.tensor([[[4.505]]]), atol=1e-6)
    assert torch.allclose(runner.rollout_data["next_values"], torch.tensor([[[4.0]]]), atol=1e-6)
    assert torch.allclose(runner.rollout_data["next_embeddings"], torch.ones(1, 1, 8) * 9.0)


def test_compute_returns_uses_sampled_next_actions_for_non_truncated_steps() -> None:
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
            torch.tensor([2.0]),
            action,
            torch.tensor(1.0),
            torch.tensor(1.0),
        )

    recorded_log_prob_actions: list[torch.Tensor] = []

    class FakeDist:
        def sample(self, sample_shape: tuple[int, ...] = torch.Size()) -> torch.Tensor:
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
        anchor = next(self.parameters()).sum() * 0.0
        values = anchor + torch.ones(value_shape, dtype=torch.float32)
        logits = anchor + torch.zeros(*value_shape, self.num_atoms, dtype=torch.float32)
        pred = anchor + torch.ones(*value_shape, 8, dtype=torch.float32)
        embeddings = anchor + torch.ones(*value_shape, 8, dtype=torch.float32)
        return values, logits, pred, embeddings

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

    sampled_next_actions = recorded_log_prob_actions[0]
    assert torch.allclose(sampled_next_actions[0], torch.tensor([[0.31, 0.32]]), atol=1e-6)
    assert torch.allclose(sampled_next_actions[1], torch.tensor([[0.41, 0.42]]), atol=1e-6)


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


def test_reppo_runner_rejects_unsupported_policy_keys() -> None:
    env = DummyEnv()
    cfg = {
        "num_steps_per_env": 1,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "policy": {
            "class_name": "ReppoPolicy",
            "critic_class_name": "ReppoCritic",
            "actor_hidden_dims": [8],
            "critic_hidden_dims": [8],
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "bogus_key": 123,
        },
        "algorithm": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "num_mini_batches": 1,
            "num_learning_epochs": 1,
        },
    }

    with mock.patch("rsl_rl.runners.reppo_runner.Logger"):
        try:
            ReppoRunner(env, cfg, log_dir=None, device="cpu")
        except ValueError as exc:
            assert "bogus_key" in str(exc)
        else:
            raise AssertionError("Expected unsupported REPPO config keys to raise ValueError")


def test_reppo_normalization_updates_include_next_observations() -> None:
    runner = _make_runner(num_steps_per_env=1)
    runner.policy.actor_obs_normalization = True
    runner.critic.critic_obs_normalization = True

    actor_updates: list[torch.Tensor] = []
    critic_updates: list[torch.Tensor] = []
    runner.policy.actor_obs_normalizer.update = lambda x: actor_updates.append(x.clone())  # type: ignore[method-assign]
    runner.critic.critic_obs_normalizer.update = lambda x: critic_updates.append(x.clone())  # type: ignore[method-assign]

    obs = runner.env.get_observations()
    runner.act(obs)
    next_obs = TensorDict(
        {
            "policy": torch.tensor([[5.0, 6.0]], dtype=torch.float32),
            "critic": torch.tensor([[7.0, 8.0]], dtype=torch.float32),
        },
        batch_size=[1],
    )
    runner.process_env_step(next_obs, rewards=torch.tensor([1.0]), dones=torch.tensor([0.0]), extras={})
    runner.compute_returns(next_obs)

    assert len(actor_updates) == 1
    assert len(critic_updates) == 1
    assert torch.allclose(actor_updates[0], torch.tensor([[1.0, 2.0], [5.0, 6.0]]))
    assert torch.allclose(critic_updates[0], torch.tensor([[3.0, 4.0], [7.0, 8.0]]))


def test_reppo_update_uses_policy_entropy_for_loss_and_logs_base_entropy() -> None:
    runner = _make_runner(num_steps_per_env=1)
    runner.rollout_data = {
        "observations": torch.zeros(1, 1, 2),
        "critic_observations": torch.zeros(1, 1, 2),
        "actions": torch.zeros(1, 1, 2),
        "rewards": torch.zeros(1, 1, 1),
        "dones": torch.zeros(1, 1, 1),
        "truncations": torch.zeros(1, 1, 1),
        "next_values": torch.zeros(1, 1, 1),
        "next_embeddings": torch.zeros(1, 1, 8),
        "gve": torch.zeros(1, 1, 1),
    }
    runner.rollout_metrics = {}

    def fake_sample_actions_from_normalized(self, normalized_obs: torch.Tensor):
        batch = normalized_obs.shape[0]
        anchor = next(self.parameters()).sum() * 0.0
        return (
            anchor + torch.zeros(batch, 2),
            anchor + torch.zeros(batch),
            anchor + torch.full((batch,), 3.0),
            anchor + torch.full((batch,), 5.0),
            anchor + torch.zeros(batch, 2),
            anchor + torch.tensor(1.0),
            anchor + torch.tensor(1.0),
        )

    def fake_build_distribution_from_normalized(self, normalized_obs: torch.Tensor):
        class FakeDist:
            def sample(self, sample_shape: tuple[int, ...] = torch.Size()) -> torch.Tensor:
                return torch.zeros(*sample_shape, normalized_obs.shape[0], 2)

            def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
                return torch.zeros(actions.shape[:-1], dtype=torch.float32)

        return FakeDist()

    def fake_forward_normalized(self, obs: torch.Tensor, action: torch.Tensor):
        batch_shape = action.shape[:-1]
        anchor = next(self.parameters()).sum() * 0.0
        values = anchor + torch.zeros(batch_shape, dtype=torch.float32)
        logits = anchor + torch.zeros(*batch_shape, self.num_atoms, dtype=torch.float32)
        pred = anchor + torch.zeros(*batch_shape, 8, dtype=torch.float32)
        embed = anchor + torch.zeros(*batch_shape, 8, dtype=torch.float32)
        return values, logits, pred, embed

    runner.policy.sample_actions_from_normalized = types.MethodType(fake_sample_actions_from_normalized, runner.policy)
    runner.policy.build_distribution_from_normalized = types.MethodType(
        fake_build_distribution_from_normalized, runner.policy
    )
    runner.critic.forward_normalized = types.MethodType(fake_forward_normalized, runner.critic)

    loss_dict, metric_dict = runner.update()

    assert loss_dict["entropy"] == 3.0
    assert loss_dict["actor"] > 1.0
    assert metric_dict["policy_entropy"] == 3.0
    assert metric_dict["base_entropy"] == 5.0


def test_reppo_actor_step_freezes_critic_parameters_but_keeps_action_gradient() -> None:
    runner = _make_runner(num_steps_per_env=1)
    batch_critic_obs = torch.zeros(1, 2, dtype=torch.float32)
    actions = torch.zeros(1, 2, dtype=torch.float32, requires_grad=True)

    for param in runner.critic.parameters():
        param.grad = None

    with runner._freeze_module_params(runner.critic):
        qf, _, _, _ = runner.critic.forward_normalized(batch_critic_obs, actions)
        qf.sum().backward()

    assert actions.grad is not None
    assert actions.grad.abs().sum().item() > 0.0
    assert all(param.grad is None for param in runner.critic.parameters())


def test_reppo_runner_clips_actions_for_log_prob() -> None:
    runner = _make_runner(num_steps_per_env=2)
    obs = runner.env.get_observations()

    def fake_sample_actions_from_normalized(self, normalized_obs: torch.Tensor):
        anchor = next(self.parameters()).sum() * 0.0
        action = anchor + torch.tensor([[2.5, -2.5]], dtype=torch.float32).expand(normalized_obs.shape[0], -1).clone()
        return (
            action,
            anchor + torch.zeros(normalized_obs.shape[0]),
            anchor + torch.zeros(normalized_obs.shape[0]),
            anchor + torch.full((normalized_obs.shape[0],), 2.0),
            action,
            anchor + torch.tensor(1.0),
            anchor + torch.tensor(1.0),
        )

    recorded_log_prob_actions: list[torch.Tensor] = []

    class FakeDist:
        def sample(self, sample_shape: tuple[int, ...] = torch.Size()) -> torch.Tensor:
            if sample_shape:
                return torch.tensor(
                    [
                        [[[3.0, -3.0]], [[4.0, -4.0]]],
                    ],
                    dtype=torch.float32,
                )
            return torch.tensor(
                [
                    [[5.0, -5.0]],
                    [[6.0, -6.0]],
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
        anchor = next(self.parameters()).sum() * 0.0
        values = anchor + torch.ones(value_shape, dtype=torch.float32)
        logits = anchor + torch.zeros(*value_shape, self.num_atoms, dtype=torch.float32)
        pred = anchor + torch.ones(*value_shape, 8, dtype=torch.float32)
        embeddings = anchor + torch.ones(*value_shape, 8, dtype=torch.float32)
        return values, logits, pred, embeddings

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
    runner.update()

    assert recorded_log_prob_actions
    assert all(action.abs().max().item() <= 1.0 for action in recorded_log_prob_actions)


def test_reppo_runner_load_resumes_from_next_iteration() -> None:
    runner = _make_runner()
    runner.current_learning_iteration = 7

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "model_7.pt")
        runner.save(checkpoint_path)

        resumed_runner = _make_runner()
        resumed_runner.load(checkpoint_path)

    assert resumed_runner.current_learning_iteration == 8


def test_reppo_runner_load_can_reset_global_std_to_init() -> None:
    runner = _make_runner(
        policy_overrides={
            "state_dependent_std": False,
            "reset_global_std_on_resume": True,
            "init_noise_std": 1.0,
            "actor_min_std": 0.1,
        }
    )
    output_layer = runner.policy._get_actor_output_layer()

    with torch.no_grad():
        output_layer.bias[runner.policy.num_actions :].fill_(torch.log(torch.tensor(0.2)))

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "model_1.pt")
        runner.save(checkpoint_path)

        resumed_runner = _make_runner(
            policy_overrides={
                "state_dependent_std": False,
                "reset_global_std_on_resume": True,
                "init_noise_std": 1.0,
                "actor_min_std": 0.1,
            }
        )
        resumed_runner.load(checkpoint_path)

    assert torch.allclose(resumed_runner.policy.output_std, torch.ones(2), atol=1e-6)
    assert torch.allclose(resumed_runner.policy_old.output_std, torch.ones(2), atol=1e-6)


def test_reppo_soft_bonus_matches_dt_scaled_rewards() -> None:
    runner = _make_runner(num_steps_per_env=1)
    runner.env.cfg["scale_rewards_by_dt"] = True
    runner.reward_scale = runner._resolve_reward_scale()

    data = {
        "next_observations": torch.zeros(1, 1, 2),
        "next_critic_observations": torch.zeros(1, 1, 2),
        "rewards": torch.zeros(1, 1, 1),
        "actions": torch.zeros(1, 1, 2),
        "truncations": torch.ones(1, 1, 1),
    }

    class FakeDist:
        def sample(self, sample_shape: tuple[int, ...] = torch.Size()) -> torch.Tensor:
            return torch.full((1, 1, 2), 0.25, dtype=torch.float32)

        def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
            return torch.full(actions.shape, -10.0, dtype=torch.float32)

    def fake_build_distribution_from_normalized(self, normalized_obs: torch.Tensor):
        return FakeDist()

    def fake_forward_normalized(self, obs: torch.Tensor, action: torch.Tensor):
        value_shape = action.shape[:-1]
        anchor = next(self.parameters()).sum() * 0.0
        values = anchor + torch.ones(value_shape, dtype=torch.float32)
        logits = anchor + torch.zeros(*value_shape, self.num_atoms, dtype=torch.float32)
        pred = anchor + torch.ones(*value_shape, 8, dtype=torch.float32)
        embeddings = anchor + torch.ones(*value_shape, 8, dtype=torch.float32)
        return values, logits, pred, embeddings

    runner.policy.build_distribution_from_normalized = types.MethodType(
        fake_build_distribution_from_normalized, runner.policy
    )
    runner.critic.forward_normalized = types.MethodType(fake_forward_normalized, runner.critic)
    with torch.no_grad():
        runner.policy.log_temp.fill_(0.0)

    extras = runner._compute_rollout_extras(data)

    expected_bonus = 0.99 * 20.0 * 0.02
    assert torch.allclose(extras["soft_bonus"], torch.full((1, 1, 1), expected_bonus), atol=1e-6)
    assert torch.allclose(extras["rewards"], torch.full((1, 1, 1), expected_bonus), atol=1e-6)
