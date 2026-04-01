# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, MLP


class FastTD3Actor(nn.Module):
    """Deterministic actor wrapper for FastTD3-style training."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        action_dim: int,
        num_envs: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "relu",
        obs_normalization: bool = False,
        init_scale: float = 0.01,
        std_min: float = 0.05,
        std_max: float = 0.8,
    ) -> None:
        super().__init__()
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.obs_normalization = obs_normalization
        self.n_envs = num_envs
        self.std_min = float(std_min)
        self.std_max = float(std_max)
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()
        self.mlp = MLP(self.obs_dim, action_dim, hidden_dims, activation)
        last_linear = self.mlp[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.normal_(last_linear.weight, 0.0, init_scale)
            nn.init.zeros_(last_linear.bias)
        noise_scales = (
            torch.rand(num_envs, 1) * (self.std_max - self.std_min) + self.std_min
        )
        self.register_buffer("noise_scales", noise_scales)

    def forward(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        return torch.tanh(self.mlp(actor_obs))

    def explore(
        self, obs: TensorDict, dones: torch.Tensor | None = None, deterministic: bool = False
    ) -> torch.Tensor:
        if dones is not None and torch.any(dones).item():
            new_scales = (
                torch.rand(self.n_envs, 1, device=self.noise_scales.device)
                * (self.std_max - self.std_min)
                + self.std_min
            )
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales.copy_(torch.where(dones_view, new_scales, self.noise_scales))

        actions = self(obs)
        if deterministic:
            return actions
        return (actions + torch.randn_like(actions) * self.noise_scales).clamp(-1.0, 1.0)

    def as_jit(self) -> nn.Module:
        return _TorchFastTD3Actor(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxFastTD3Actor(self, verbose)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        actor_obs = torch.cat(obs_list, dim=-1)
        return self.obs_normalizer(actor_obs)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            actor_obs = torch.cat(obs_list, dim=-1)
            self.obs_normalizer.update(actor_obs)  # type: ignore[attr-defined]

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The FastTD3 actor only supports 1D observations, got shape {obs[obs_group].shape} for "
                    f"'{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim


class _TorchFastTD3Actor(nn.Module):
    """TorchScript-friendly deterministic actor export."""

    def __init__(self, model: FastTD3Actor) -> None:
        super().__init__()
        self.obs_normalizer = nn.Identity() if not model.obs_normalization else copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(self.obs_normalizer(obs)))

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxFastTD3Actor(nn.Module):
    """ONNX export wrapper for a deterministic FastTD3 actor."""

    is_recurrent: bool = False

    def __init__(self, model: FastTD3Actor, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = nn.Identity() if not model.obs_normalization else copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.input_size = model.obs_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(self.obs_normalizer(obs)))

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class FastTD3Critic(nn.Module):
    """Distributional critic wrapper for FastTD3-style off-policy updates."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        action_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (1024, 512, 256),
        activation: str = "relu",
        obs_normalization: bool = False,
        num_atoms: int = 101,
        v_min: float = -250.0,
        v_max: float = 250.0,
    ) -> None:
        super().__init__()
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.obs_normalization = obs_normalization
        self.num_atoms = int(num_atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()
        self.mlp = MLP(self.obs_dim + action_dim, self.num_atoms, hidden_dims, activation)
        self.register_buffer("q_support", torch.linspace(self.v_min, self.v_max, self.num_atoms))

    def forward(self, obs: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        x = torch.cat([critic_obs, actions], dim=-1)
        return self.mlp(x)

    def projection(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        rewards = rewards.reshape(-1)
        bootstrap = bootstrap.reshape(-1)
        discount = discount.reshape(-1)
        batch_size = rewards.shape[0]

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * self.q_support
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        low = torch.floor(b).long()
        u = torch.ceil(b).long()

        l_mask = torch.logical_and((u > 0), (low == u))
        u_mask = torch.logical_and((low < (self.num_atoms - 1)), (low == u))

        low = torch.where(l_mask, low - 1, low)
        u = torch.where(u_mask, u + 1, u)

        next_dist = F.softmax(self.forward(obs, actions), dim=-1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=next_dist.device)
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )
        proj_dist.view(-1).index_add_(0, (low + offset).reshape(-1), (next_dist * (u.float() - b)).reshape(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).reshape(-1), (next_dist * (b - low.float())).reshape(-1))
        return proj_dist

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        return torch.sum(probs * self.q_support, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        critic_obs = torch.cat(obs_list, dim=-1)
        return self.obs_normalizer(critic_obs)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            critic_obs = torch.cat(obs_list, dim=-1)
            self.obs_normalizer.update(critic_obs)  # type: ignore[attr-defined]

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The FastTD3 critic only supports 1D observations, got shape {obs[obs_group].shape} for "
                    f"'{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim
