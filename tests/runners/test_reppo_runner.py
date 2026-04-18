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
from rsl_rl.modules import ActorQ
from rsl_rl.runners import ReppoRunner


class DummyEnv(VecEnv):
    def __init__(self) -> None:
        self.num_envs = 2
        self.num_actions = 2
        self.max_episode_length = 8
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {"scale_rewards_by_dt": False}
        self.unwrapped = types.SimpleNamespace(common_step_counter=0, step_dt=0.02)

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {
                "policy": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
                "critic": torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
            },
            batch_size=[self.num_envs],
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        del actions
        return (
            self.get_observations(),
            torch.zeros(self.num_envs, device="cpu"),
            torch.zeros(self.num_envs, device="cpu"),
            {},
        )


def _make_runner() -> ReppoRunner:
    return ReppoRunner(
        DummyEnv(),
        {
            "num_steps_per_env": 2,
            "save_interval": 100,
            "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
            "policy": {
                "class_name": "ReppoPolicy",
                "critic_class_name": "ReppoCritic",
                "actor_obs_normalization": False,
                "critic_obs_normalization": False,
                "actor_hidden_dims": (16, 8),
                "critic_hidden_dims": (16, 8),
                "ent_start": 0.001,
                "kl_start": 0.01,
                "state_dependent_std": False,
                "noise_std_type": "scalar",
            },
            "algorithm": {
                "class_name": "Reppo",
                "num_learning_epochs": 1,
                "num_mini_batches": 1,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "lam": 0.95,
                "num_atoms": 151,
                "vmin": -10.0,
                "vmax": 30.0,
                "kl_bound": 0.1,
                "ent_target_mult": 0.5,
            },
        },
        log_dir=None,
        device="cpu",
    )


def test_reppo_runner_translates_legacy_config_to_official_actor_q() -> None:
    runner = _make_runner()

    assert isinstance(runner.alg.policy, ActorQ)
    assert list(runner.cfg["obs_groups"]["policy"]) == ["policy"]
    assert runner.alg.desired_kl == 0.1
    assert runner.alg.target_entropy == -1.0
    assert runner.alg.policy.num_critic_bins == 151
    assert runner.alg.policy.vmin == -10.0
    assert runner.alg.policy.vmax == 30.0


def test_reppo_runner_returns_callable_inference_policy() -> None:
    runner = _make_runner()

    inference_policy = runner.get_inference_policy(device="cpu")
    actions = inference_policy(runner.env.get_observations())

    assert callable(inference_policy)
    assert actions.shape == (runner.env.num_envs, runner.env.num_actions)


def test_reppo_runner_save_and_load_round_trip() -> None:
    runner = _make_runner()
    runner.current_learning_iteration = 12

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        runner.save(path, infos={"note": "test"})

        resumed_runner = _make_runner()
        infos = resumed_runner.load(path, map_location="cpu")

    assert infos == {"note": "test"}
    assert resumed_runner.current_learning_iteration == 12
    for key, value in runner.alg.policy.state_dict().items():
        assert torch.equal(value, resumed_runner.alg.policy.state_dict()[key])


def test_reppo_runner_initializes_distributed_process_group() -> None:
    cfg = {
        "num_steps_per_env": 2,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "policy": {
            "class_name": "ReppoPolicy",
            "critic_class_name": "ReppoCritic",
            "actor_hidden_dims": (8,),
            "critic_hidden_dims": (8,),
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
        },
        "algorithm": {
            "class_name": "Reppo",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
        },
    }

    with (
        mock.patch.dict(os.environ, {"WORLD_SIZE": "2", "LOCAL_RANK": "1", "RANK": "1"}, clear=False),
        mock.patch("torch.distributed.init_process_group") as init_pg,
        mock.patch("torch.cuda.set_device") as set_device,
        mock.patch.object(
            ReppoRunner,
            "_construct_algorithm",
            return_value=types.SimpleNamespace(policy=mock.Mock()),
        ),
        mock.patch("rsl_rl.runners.reppo_runner.Logger"),
    ):
        runner = ReppoRunner(DummyEnv(), cfg, log_dir=None, device="cuda:1")

    assert runner.is_distributed is True
    assert runner.multi_gpu_cfg == {"global_rank": 1, "local_rank": 1, "world_size": 2}
    init_pg.assert_called_once_with(backend="nccl", rank=1, world_size=2)
    set_device.assert_called_once_with(1)
