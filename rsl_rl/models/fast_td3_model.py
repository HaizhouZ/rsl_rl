# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

import torch
import torch.nn as nn
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
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
    ) -> None:
        super().__init__()
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()
        self.mlp = MLP(self.obs_dim, action_dim, hidden_dims, activation)

    def forward(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        return torch.tanh(self.mlp(actor_obs))

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
    """MLP critic wrapper for FastTD3-style off-policy updates."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        action_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
    ) -> None:
        super().__init__()
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()
        self.mlp = MLP(self.obs_dim + action_dim, 1, hidden_dims, activation)

    def forward(self, obs: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        x = torch.cat([critic_obs, actions], dim=-1)
        return self.mlp(x)

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
