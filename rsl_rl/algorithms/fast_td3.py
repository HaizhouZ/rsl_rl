# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import FastTD3Actor, FastTD3Critic
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.storage import TensorDictReplayBuffer
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


class RewardNormalizer(nn.Module):
    """Reward scale normalizer matching the reference FastTD3 behavior."""

    def __init__(
        self,
        num_envs: int,
        gamma: float,
        device: str,
        g_max: float = 10.0,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.register_buffer("G", torch.zeros(num_envs, device=device))
        self.register_buffer("G_r_max", torch.zeros(1, device=device))
        self.G_rms = EmpiricalNormalization(shape=1).to(device)
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon

    def update_stats(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        self.G.copy_(self.gamma * (1.0 - dones) * self.G + rewards)
        self.G_rms.update(self.G.view(-1, 1))
        self.G_r_max.copy_(torch.maximum(self.G_r_max, self.G.abs().max().view(1)))

    def forward(self, rewards: torch.Tensor) -> torch.Tensor:
        var_denominator = self.G_rms.std[0] + self.epsilon
        min_required_denominator = self.G_r_max / self.g_max
        denominator = torch.maximum(var_denominator, min_required_denominator)
        return rewards / denominator


class FastTD3:
    """Off-policy FastTD3-style algorithm scaffold."""

    actor: FastTD3Actor

    def __init__(
        self,
        actor: FastTD3Actor,
        critic1: FastTD3Critic,
        critic2: FastTD3Critic,
        replay_buffer: TensorDictReplayBuffer,
        *,
        critic1_target: FastTD3Critic,
        critic2_target: FastTD3Critic,
        batch_size: int = 256,
        learning_starts: int = 1_000,
        num_updates: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_frequency: int = 2,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        actor_learning_rate: float = 3e-4,
        actor_learning_rate_end: float | None = None,
        critic_learning_rate: float = 3e-4,
        critic_learning_rate_end: float | None = None,
        weight_decay: float = 0.1,
        max_grad_norm: float = 1.0,
        optimizer: str = "adamw",
        use_cdq: bool = True,
        reward_normalization: bool = False,
        scheduler_steps: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.actor = actor.to(self.device)
        self.critic1 = critic1.to(self.device)
        self.critic2 = critic2.to(self.device)
        self.policy = self.actor
        self.critic = self.critic1
        self.critic1_target = critic1_target.to(self.device)
        self.critic2_target = critic2_target.to(self.device)
        self.replay_buffer = replay_buffer

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_target.eval()
        self.critic2_target.eval()

        optimizer_cls = resolve_optimizer(optimizer)
        self.actor_optimizer = optimizer_cls(
            self.actor.parameters(), lr=actor_learning_rate, weight_decay=weight_decay
        )
        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_optimizer = optimizer_cls(
            critic_params, lr=critic_learning_rate, weight_decay=weight_decay
        )
        self.actor_scheduler = None
        self.critic_scheduler = None
        if scheduler_steps is not None and scheduler_steps > 0:
            actor_lr_end = actor_learning_rate if actor_learning_rate_end is None else actor_learning_rate_end
            critic_lr_end = critic_learning_rate if critic_learning_rate_end is None else critic_learning_rate_end
            self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.actor_optimizer,
                T_max=scheduler_steps,
                eta_min=actor_lr_end,
            )
            self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.critic_optimizer,
                T_max=scheduler_steps,
                eta_min=critic_lr_end,
            )

        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.num_updates = num_updates
        self.gamma = gamma
        self.tau = tau
        self.policy_frequency = policy_frequency
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.max_grad_norm = max_grad_norm
        self.learning_rate = critic_learning_rate
        self.use_cdq = use_cdq
        self.reward_normalizer = (
            RewardNormalizer(
                self.actor.n_envs,
                gamma=self.gamma,
                device=self.device,
                g_max=min(abs(self.critic1.v_min), abs(self.critic1.v_max)),
            )
            if reward_normalization
            else None
        )
        self.update_step = 0
        self.pending_transition: TensorDict | None = None

    def act(self, obs: TensorDict, dones: torch.Tensor | None = None) -> torch.Tensor:
        with torch.inference_mode():
            if self.actor.training:
                actions = self.actor.explore(obs, dones=dones)
            else:
                actions = self.actor(obs)
        self.pending_transition = TensorDict(
            {"observations": obs.clone(), "actions": actions.clone()}, batch_size=obs.batch_size
        )
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        if self.pending_transition is None:
            raise RuntimeError("process_env_step called before act")

        self.actor.update_normalization(self.pending_transition["observations"])
        self.critic1.update_normalization(self.pending_transition["observations"])
        self.critic2.update_normalization(self.pending_transition["observations"])
        if self.reward_normalizer is not None:
            self.reward_normalizer.update_stats(rewards, dones.float())

        truncations = extras.get("time_outs")
        if truncations is None:
            truncations = torch.zeros_like(dones)

        transition = TensorDict(
            {
                "observations": self.pending_transition["observations"].clone(),
                "actions": self.pending_transition["actions"].clone(),
                "next": TensorDict(
                    {
                        "observations": obs.clone(),
                        "rewards": rewards.clone().unsqueeze(-1).to(dtype=torch.float32),
                        "dones": dones.clone().unsqueeze(-1).to(dtype=torch.float32),
                        "truncations": truncations.clone().unsqueeze(-1).to(dtype=torch.float32),
                        "effective_n_steps": torch.ones_like(rewards, dtype=torch.float32).unsqueeze(-1),
                    },
                    batch_size=obs.batch_size,
                ),
            },
            batch_size=obs.batch_size,
        )
        self.replay_buffer.add(transition)
        self.pending_transition = None

    def compute_returns(self, obs: TensorDict) -> None:
        """FastTD3 is off-policy and does not use return bootstrapping here."""

    def update(self) -> dict[str, float]:
        per_env_batch = max(1, self.batch_size // self.actor.n_envs)
        if len(self.replay_buffer) < max(self.learning_starts, per_env_batch):
            return {}

        logs: dict[str, list[float]] = defaultdict(list)
        for _ in range(self.num_updates):
            batch = self.replay_buffer.sample(per_env_batch, device=self.device)
            actor_obs = batch["observations"]
            next_obs = batch["next"]["observations"]
            actions = batch["actions"]
            rewards = batch["next"]["rewards"]
            dones = batch["next"]["dones"]
            truncations = batch["next"]["truncations"]
            effective_n_steps = batch["next"].get(
                "effective_n_steps", torch.ones_like(rewards, dtype=torch.float32)
            )
            if self.reward_normalizer is not None:
                rewards = self.reward_normalizer(rewards)
            bootstrap = torch.logical_or(~dones.bool(), truncations.bool()).float()
            discount = torch.full_like(effective_n_steps, self.gamma, dtype=torch.float32).pow(
                effective_n_steps.to(dtype=torch.float32)
            )

            with torch.no_grad():
                target_actions = self.actor(next_obs)
                if self.target_noise > 0:
                    noise = (torch.randn_like(target_actions) * self.target_noise).clamp(
                        -self.noise_clip, self.noise_clip
                    )
                    target_actions = (target_actions + noise).clamp(-1.0, 1.0)
                target_q1 = self.critic1_target.projection(
                    next_obs, target_actions, rewards, bootstrap, discount
                )
                target_q2 = self.critic2_target.projection(
                    next_obs, target_actions, rewards, bootstrap, discount
                )
                target_q1_value = self.critic1_target.get_value(target_q1)
                target_q2_value = self.critic2_target.get_value(target_q2)
                if self.use_cdq:
                    target_q = torch.where(
                        target_q1_value.unsqueeze(-1) < target_q2_value.unsqueeze(-1),
                        target_q1,
                        target_q2,
                    )
                    target_q1 = target_q2 = target_q

            q1 = self.critic1(actor_obs, actions)
            q2 = self.critic2(actor_obs, actions)
            q1_loss = -(target_q1 * F.log_softmax(q1, dim=-1)).sum(dim=-1).mean()
            q2_loss = -(target_q2 * F.log_softmax(q2, dim=-1)).sum(dim=-1).mean()
            critic_loss = q1_loss + q2_loss

            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.critic1.parameters()) + list(self.critic2.parameters()),
                    self.max_grad_norm,
                )
            self.critic_optimizer.step()
            if self.critic_scheduler is not None:
                self.critic_scheduler.step()

            logs["critic_loss"].append(float(critic_loss.detach()))
            logs["q1_mean"].append(float(self.critic1.get_value(F.softmax(q1, dim=-1)).mean().detach()))
            logs["q2_mean"].append(float(self.critic2.get_value(F.softmax(q2, dim=-1)).mean().detach()))

            if self.update_step % self.policy_frequency == 0:
                actor_actions = self.actor(actor_obs)
                q1_actor = self.critic1(actor_obs, actor_actions)
                q2_actor = self.critic2(actor_obs, actor_actions)
                q1_actor_value = self.critic1.get_value(F.softmax(q1_actor, dim=-1))
                q2_actor_value = self.critic2.get_value(F.softmax(q2_actor, dim=-1))
                if self.use_cdq:
                    actor_value = torch.minimum(q1_actor_value, q2_actor_value)
                else:
                    actor_value = (q1_actor_value + q2_actor_value) / 2.0
                actor_loss = -actor_value.mean()
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                if self.actor_scheduler is not None:
                    self.actor_scheduler.step()
                logs["actor_loss"].append(float(actor_loss.detach()))

                self._soft_update(self.critic1_target, self.critic1, self.tau)
                self._soft_update(self.critic2_target, self.critic2, self.tau)
            else:
                self._soft_update(self.critic1_target, self.critic1, self.tau)
                self._soft_update(self.critic2_target, self.critic2, self.tau)

            self.update_step += 1
            self.learning_rate = self.critic_optimizer.param_groups[0]["lr"]

        return {key: sum(values) / len(values) for key, values in logs.items() if values}

    def train_mode(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        if self.reward_normalizer is not None:
            self.reward_normalizer.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        if self.reward_normalizer is not None:
            self.reward_normalizer.eval()

    def save(self) -> dict:
        return self._clone_checkpoint_tensors(
            {
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "critic1_target_state_dict": self.critic1_target.state_dict(),
            "critic2_target_state_dict": self.critic2_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "replay_buffer_state_dict": self.replay_buffer.state_dict(),
            "update_step": self.update_step,
            **(
                {"reward_normalizer_state_dict": self.reward_normalizer.state_dict()}
                if self.reward_normalizer is not None
                else {}
            ),
            }
        )

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_cfg = load_cfg or {}
        if load_cfg.get("actor", True):
            if "actor_state_dict" not in loaded_dict:
                raise KeyError("FastTD3 checkpoint is missing 'actor_state_dict'.")
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic1", True):
            if "critic1_state_dict" not in loaded_dict:
                raise KeyError("FastTD3 checkpoint is missing 'critic1_state_dict'.")
            self.critic1.load_state_dict(loaded_dict["critic1_state_dict"], strict=strict)
        if load_cfg.get("critic2", True):
            if "critic2_state_dict" not in loaded_dict:
                raise KeyError("FastTD3 checkpoint is missing 'critic2_state_dict'.")
            self.critic2.load_state_dict(loaded_dict["critic2_state_dict"], strict=strict)
        if load_cfg.get("critic1_target", True):
            if "critic1_target_state_dict" not in loaded_dict:
                raise KeyError("FastTD3 checkpoint is missing 'critic1_target_state_dict'.")
            self.critic1_target.load_state_dict(loaded_dict["critic1_target_state_dict"], strict=strict)
        if load_cfg.get("critic2_target", True):
            if "critic2_target_state_dict" not in loaded_dict:
                raise KeyError("FastTD3 checkpoint is missing 'critic2_target_state_dict'.")
            self.critic2_target.load_state_dict(loaded_dict["critic2_target_state_dict"], strict=strict)
        if load_cfg.get("actor_optimizer", True):
            if "actor_optimizer_state_dict" not in loaded_dict:
                raise KeyError("FastTD3 checkpoint is missing 'actor_optimizer_state_dict'.")
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
        if load_cfg.get("critic_optimizer", True):
            if "critic_optimizer_state_dict" not in loaded_dict:
                raise KeyError("FastTD3 checkpoint is missing 'critic_optimizer_state_dict'.")
            self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
        if load_cfg.get("replay_buffer", True):
            if "replay_buffer_state_dict" not in loaded_dict:
                raise KeyError("FastTD3 checkpoint is missing 'replay_buffer_state_dict'.")
            self.replay_buffer.load_state_dict(loaded_dict["replay_buffer_state_dict"])
        if self.reward_normalizer is not None and load_cfg.get("reward_normalizer", True):
            if "reward_normalizer_state_dict" not in loaded_dict:
                raise KeyError("FastTD3 checkpoint is missing 'reward_normalizer_state_dict'.")
            self.reward_normalizer.load_state_dict(loaded_dict["reward_normalizer_state_dict"])
        if load_cfg.get("iteration", True):
            if "update_step" not in loaded_dict:
                raise KeyError("FastTD3 checkpoint is missing 'update_step'.")
            self.update_step = int(loaded_dict["update_step"])
        return bool(load_cfg.get("iteration", True))

    def get_policy(self) -> FastTD3Actor:
        return self.actor

    def broadcast_parameters(self) -> None:
        """Compatibility no-op for the current single-process setup."""

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> FastTD3:
        alg_class: type[FastTD3] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore[arg-type]
        actor_class: type[FastTD3Actor] = resolve_callable(
            cfg["actor"].pop("class_name", "FastTD3Actor")
        )  # type: ignore[arg-type]
        critic_class: type[FastTD3Critic] = resolve_callable(
            cfg["critic"].pop("class_name", "FastTD3Critic")
        )  # type: ignore[arg-type]

        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        algorithm_cfg = dict(cfg["algorithm"])
        replay_size = algorithm_cfg.pop("replay_size", None)
        if replay_size is None:
            replay_size = algorithm_cfg.pop("buffer_size", 100_000)
        else:
            algorithm_cfg.pop("buffer_size", None)

        legacy_learning_rate = algorithm_cfg.pop("learning_rate", 3e-4)
        actor_learning_rate = algorithm_cfg.pop("actor_learning_rate", legacy_learning_rate)
        actor_learning_rate_end = algorithm_cfg.pop("actor_learning_rate_end", actor_learning_rate)
        critic_learning_rate = algorithm_cfg.pop("critic_learning_rate", actor_learning_rate)
        critic_learning_rate_end = algorithm_cfg.pop("critic_learning_rate_end", critic_learning_rate)
        weight_decay = algorithm_cfg.pop("weight_decay", 0.1)
        policy_frequency = algorithm_cfg.pop("policy_frequency", algorithm_cfg.pop("policy_delay", 2))
        use_cdq = algorithm_cfg.pop("use_cdq", True)
        reward_normalization = algorithm_cfg.pop("reward_normalization", False)
        n_steps = algorithm_cfg.pop("n_steps", 1)
        num_atoms = algorithm_cfg.pop("num_atoms", 101)
        v_min = algorithm_cfg.pop("v_min", -250.0)
        v_max = algorithm_cfg.pop("v_max", 250.0)
        algorithm_cfg.pop("exploration_noise", None)

        actor_kwargs = dict(cfg["actor"])
        actor_kwargs.setdefault("init_scale", 0.01)
        actor_kwargs.setdefault("std_min", 0.05)
        actor_kwargs.setdefault("std_max", 0.8)
        actor: FastTD3Actor = actor_class(
            obs,
            cfg["obs_groups"],
            "actor",
            env.num_actions,
            env.num_envs,
            **actor_kwargs,
        ).to(device)
        critic_kwargs = dict(cfg["critic"])
        critic_kwargs.setdefault("num_atoms", num_atoms)
        critic_kwargs.setdefault("v_min", v_min)
        critic_kwargs.setdefault("v_max", v_max)
        critic1: FastTD3Critic = critic_class(
            obs,
            cfg["obs_groups"],
            "critic",
            env.num_actions,
            **critic_kwargs,
        ).to(device)
        critic2: FastTD3Critic = critic_class(
            obs,
            cfg["obs_groups"],
            "critic",
            env.num_actions,
            **copy.deepcopy(critic_kwargs),
        ).to(device)

        replay_buffer = TensorDictReplayBuffer(
            replay_size,
            device="cpu",
            num_envs=env.num_envs,
            n_steps=n_steps,
            gamma=algorithm_cfg.get("gamma", 0.99),
        )

        algorithm_cfg.setdefault("batch_size", min(256, replay_size))
        algorithm_cfg.setdefault("learning_starts", min(1_000, replay_size))
        algorithm_cfg.setdefault("num_updates", 1)
        algorithm_cfg.setdefault("gamma", 0.99)
        algorithm_cfg.setdefault("tau", 0.005)
        algorithm_cfg.setdefault("target_noise", 0.2)
        algorithm_cfg.setdefault("noise_clip", 0.5)
        algorithm_cfg.setdefault("policy_frequency", policy_frequency)
        algorithm_cfg.setdefault("actor_learning_rate", actor_learning_rate)
        algorithm_cfg.setdefault("actor_learning_rate_end", actor_learning_rate_end)
        algorithm_cfg.setdefault("critic_learning_rate", critic_learning_rate)
        algorithm_cfg.setdefault("critic_learning_rate_end", critic_learning_rate_end)
        algorithm_cfg.setdefault("weight_decay", weight_decay)
        algorithm_cfg.setdefault("max_grad_norm", 1.0)
        algorithm_cfg.setdefault("optimizer", "adamw")
        algorithm_cfg.setdefault("use_cdq", use_cdq)
        algorithm_cfg.setdefault("reward_normalization", reward_normalization)
        algorithm_cfg.setdefault(
            "scheduler_steps",
            max(1, int(cfg.get("max_iterations", 1)) * int(algorithm_cfg.get("num_updates", 1))),
        )
        algorithm_cfg.pop("rnd_cfg", None)
        algorithm_cfg.pop("symmetry_cfg", None)
        algorithm_cfg.pop("multi_gpu_cfg", None)

        return alg_class(
            actor,
            critic1,
            critic2,
            replay_buffer,
            critic1_target=copy.deepcopy(critic1),
            critic2_target=copy.deepcopy(critic2),
            device=device,
            **algorithm_cfg,
        )

    @staticmethod
    def _soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float | None = None) -> None:
        tau = 0.005 if tau is None else tau
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * source_param.data)

    @classmethod
    def _clone_checkpoint_tensors(cls, value):
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, TensorDict):
            return value.clone()
        if isinstance(value, Mapping):
            return {key: cls._clone_checkpoint_tensors(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(cls._clone_checkpoint_tensors(item) for item in value)
        if isinstance(value, list):
            return [cls._clone_checkpoint_tensors(item) for item in value]
        return value
