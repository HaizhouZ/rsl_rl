from __future__ import annotations

import copy

import torch
import pytest
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.runners import OffPolicyRunner

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4
MAX_EP_LEN = 32


class DummyEnv(VecEnv):
    def __init__(self, device: str = "cpu") -> None:
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = MAX_EP_LEN
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {}

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {"policy": torch.randn(self.num_envs, OBS_DIM, device=self.device)},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).float()
        self.episode_length_buf[dones.bool()] = 0
        obs = self.get_observations()
        rewards = torch.randn(self.num_envs, device=self.device)
        extras = {"time_outs": torch.zeros(self.num_envs, device=self.device)}
        return obs, rewards, dones, extras


def _make_train_cfg() -> dict:
    return {
        "num_steps_per_env": 4,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "hidden_dims": [32, 32],
            "activation": "relu",
            "obs_normalization": True,
        },
        "critic": {
            "hidden_dims": [32, 32],
            "activation": "relu",
            "obs_normalization": True,
        },
        "algorithm": {
            "class_name": "FastTD3",
            "batch_size": 8,
            "learning_starts": 0,
            "num_updates": 1,
            "policy_frequency": 1,
            "target_noise": 0.1,
            "noise_clip": 0.2,
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "weight_decay": 0.0,
            "num_atoms": 51,
            "v_min": -10.0,
            "v_max": 10.0,
            "use_cdq": True,
            "reward_normalization": True,
            "replay_size": 128,
        },
    }


def test_off_policy_runner_learns_and_populates_replay_buffer() -> None:
    runner = OffPolicyRunner(DummyEnv(), _make_train_cfg(), log_dir=None, device="cpu")
    before = copy.deepcopy(runner.alg.actor.state_dict())

    runner.learn(num_learning_iterations=1)

    assert len(runner.alg.replay_buffer) > 0
    changed = any(not torch.equal(before[key], value) for key, value in runner.alg.actor.state_dict().items())
    assert changed, "Actor parameters should change after a FastTD3 learning step"
    saved = runner.alg.save()
    assert "actor_state_dict" in saved
    assert "critic1_state_dict" in saved
    assert "critic2_state_dict" in saved
    assert "critic1_target_state_dict" in saved
    assert "critic2_target_state_dict" in saved
    assert "replay_buffer_state_dict" in saved
    assert "reward_normalizer_state_dict" in saved
    assert "actor_target_state_dict" not in saved


def test_fast_td3_rejects_legacy_checkpoint_shape() -> None:
    runner = OffPolicyRunner(DummyEnv(), _make_train_cfg(), log_dir=None, device="cpu")
    legacy_checkpoint = {
        "model_state_dict": copy.deepcopy(runner.alg.actor.state_dict()),
        "optimizer_state_dict": copy.deepcopy(runner.alg.actor_optimizer.state_dict()),
        "iter": 0,
        "infos": {},
    }

    with pytest.raises(KeyError, match="actor_state_dict"):
        runner.alg.load(
            legacy_checkpoint,
        {
            "actor": True,
            "critic1": True,
            "critic2": True,
            "critic1_target": True,
            "critic2_target": True,
            "actor_optimizer": True,
            "critic_optimizer": True,
                "replay_buffer": True,
                "iteration": True,
            },
            strict=True,
        )
