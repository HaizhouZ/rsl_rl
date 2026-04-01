# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from collections import defaultdict

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import FastTD3Actor, FastTD3Critic
from rsl_rl.storage import TensorDictReplayBuffer
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


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
        actor_target: FastTD3Actor,
        critic1_target: FastTD3Critic,
        critic2_target: FastTD3Critic,
        batch_size: int = 256,
        learning_starts: int = 1_000,
        num_updates: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        exploration_noise: float = 0.1,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        learning_rate: float = 3e-4,
        max_grad_norm: float = 1.0,
        optimizer: str = "adamw",
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.actor = actor.to(self.device)
        self.critic1 = critic1.to(self.device)
        self.critic2 = critic2.to(self.device)
        self.actor_target = actor_target.to(self.device)
        self.critic1_target = critic1_target.to(self.device)
        self.critic2_target = critic2_target.to(self.device)
        self.replay_buffer = replay_buffer

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_target.eval()
        self.critic1_target.eval()
        self.critic2_target.eval()

        optimizer_cls = resolve_optimizer(optimizer)
        self.actor_optimizer = optimizer_cls(self.actor.parameters(), lr=learning_rate)
        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_optimizer = optimizer_cls(critic_params, lr=learning_rate)

        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.num_updates = num_updates
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.update_step = 0
        self.pending_transition: TensorDict | None = None

    def act(self, obs: TensorDict) -> torch.Tensor:
        with torch.inference_mode():
            actions = self.actor(obs)
            if self.actor.training and self.exploration_noise > 0:
                noise = torch.randn_like(actions) * self.exploration_noise
                actions = (actions + noise).clamp(-1.0, 1.0)
        self.pending_transition = TensorDict({"observations": obs.clone(), "actions": actions.clone()}, batch_size=obs.batch_size)
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        if self.pending_transition is None:
            raise RuntimeError("process_env_step called before act")

        self.actor.update_normalization(self.pending_transition["observations"])
        self.critic1.update_normalization(self.pending_transition["observations"])
        self.critic2.update_normalization(self.pending_transition["observations"])

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
        if len(self.replay_buffer) < max(self.learning_starts, self.batch_size):
            return {}

        logs: dict[str, list[float]] = defaultdict(list)
        for _ in range(self.num_updates):
            batch = self.replay_buffer.sample(self.batch_size, device=self.device)
            actor_obs = batch["observations"]
            next_obs = batch["next"]["observations"]
            actions = batch["actions"]
            rewards = batch["next"]["rewards"]
            dones = batch["next"]["dones"]
            truncations = batch["next"]["truncations"]
            bootstrap = torch.logical_or(~dones.bool(), truncations.bool()).float()

            with torch.no_grad():
                target_actions = self.actor_target(next_obs)
                if self.target_noise > 0:
                    noise = (torch.randn_like(target_actions) * self.target_noise).clamp(
                        -self.noise_clip, self.noise_clip
                    )
                    target_actions = (target_actions + noise).clamp(-1.0, 1.0)
                target_q1 = self.critic1_target(next_obs, target_actions)
                target_q2 = self.critic2_target(next_obs, target_actions)
                target_q = torch.minimum(target_q1, target_q2)
                target = rewards + self.gamma * bootstrap * target_q

            q1 = self.critic1(actor_obs, actions)
            q2 = self.critic2(actor_obs, actions)
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.critic1.parameters()) + list(self.critic2.parameters()),
                    self.max_grad_norm,
                )
            self.critic_optimizer.step()

            logs["critic_loss"].append(float(critic_loss.detach()))
            logs["q1_mean"].append(float(q1.mean().detach()))
            logs["q2_mean"].append(float(q2.mean().detach()))

            if self.update_step % self.policy_delay == 0:
                actor_actions = self.actor(actor_obs)
                actor_loss = -self.critic1(actor_obs, actor_actions).mean()
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                logs["actor_loss"].append(float(actor_loss.detach()))

                self._soft_update(self.actor_target, self.actor, self.tau)
                self._soft_update(self.critic1_target, self.critic1, self.tau)
                self._soft_update(self.critic2_target, self.critic2, self.tau)

            self.update_step += 1

        return {key: sum(values) / len(values) for key, values in logs.items() if values}

    def train_mode(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def save(self) -> dict:
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic1_target_state_dict": self.critic1_target.state_dict(),
            "critic2_target_state_dict": self.critic2_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "replay_buffer_state_dict": self.replay_buffer.state_dict(),
            "update_step": self.update_step,
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if "actor_state_dict" in loaded_dict:
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if "critic1_state_dict" in loaded_dict:
            self.critic1.load_state_dict(loaded_dict["critic1_state_dict"], strict=strict)
        if "critic2_state_dict" in loaded_dict:
            self.critic2.load_state_dict(loaded_dict["critic2_state_dict"], strict=strict)
        if "actor_target_state_dict" in loaded_dict:
            self.actor_target.load_state_dict(loaded_dict["actor_target_state_dict"], strict=strict)
        if "critic1_target_state_dict" in loaded_dict:
            self.critic1_target.load_state_dict(loaded_dict["critic1_target_state_dict"], strict=strict)
        if "critic2_target_state_dict" in loaded_dict:
            self.critic2_target.load_state_dict(loaded_dict["critic2_target_state_dict"], strict=strict)
        if "actor_optimizer_state_dict" in loaded_dict:
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in loaded_dict:
            self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
        if "replay_buffer_state_dict" in loaded_dict:
            self.replay_buffer.load_state_dict(loaded_dict["replay_buffer_state_dict"])
        if "update_step" in loaded_dict:
            self.update_step = int(loaded_dict["update_step"])
        return "update_step" in loaded_dict

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
        replay_buffer = TensorDictReplayBuffer(replay_size, device="cpu")

        actor: FastTD3Actor = actor_class(
            obs,
            cfg["obs_groups"],
            "actor",
            env.num_actions,
            **cfg["actor"],
        ).to(device)
        critic1: FastTD3Critic = critic_class(
            obs,
            cfg["obs_groups"],
            "critic",
            env.num_actions,
            **cfg["critic"],
        ).to(device)
        critic2: FastTD3Critic = critic_class(
            obs,
            cfg["obs_groups"],
            "critic",
            env.num_actions,
            **copy.deepcopy(cfg["critic"]),
        ).to(device)

        algorithm_cfg.setdefault("batch_size", min(256, replay_size))
        algorithm_cfg.setdefault("learning_starts", min(1_000, replay_size))
        algorithm_cfg.setdefault("num_updates", 1)
        algorithm_cfg.setdefault("gamma", 0.99)
        algorithm_cfg.setdefault("tau", 0.005)
        algorithm_cfg.setdefault("policy_delay", 2)
        algorithm_cfg.setdefault("exploration_noise", 0.1)
        algorithm_cfg.setdefault("target_noise", 0.2)
        algorithm_cfg.setdefault("noise_clip", 0.5)
        algorithm_cfg.setdefault("learning_rate", 3e-4)
        algorithm_cfg.setdefault("max_grad_norm", 1.0)
        algorithm_cfg.setdefault("optimizer", "adamw")
        algorithm_cfg.pop("rnd_cfg", None)
        algorithm_cfg.pop("symmetry_cfg", None)
        algorithm_cfg.pop("multi_gpu_cfg", None)

        return alg_class(
            actor,
            critic1,
            critic2,
            replay_buffer,
            actor_target=copy.deepcopy(actor),
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
