# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import Transform

from rsl_rl.modules import EmpiricalNormalization, MLP


class TanhTransform(Transform):
    """Numerically stable tanh transform."""

    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1
    log2 = torch.log(torch.tensor(2.0))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TanhTransform)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return x.tanh()

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.atanh(y)

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 2.0 * (self.log2 - x - torch.nn.functional.softplus(-2.0 * x))


def hl_gauss(inp: torch.Tensor, vmin: float, vmax: float, num_atoms: int) -> torch.Tensor:
    """Convert scalar targets to smoothed categorical distributions."""
    x = torch.clip(inp, vmin, max=vmax)
    bin_width = (vmax - vmin) / (num_atoms - 1)
    sigma_to_final_sigma_ratio = 0.75
    support = torch.linspace(
        vmin - bin_width / 2,
        vmax + bin_width / 2,
        num_atoms + 1,
        device=inp.device,
    )
    sigma = bin_width * sigma_to_final_sigma_ratio
    cdf_evals = torch.erf(
        (support.unsqueeze(0) - x).squeeze()
        / (torch.sqrt(torch.tensor(2.0, device=inp.device)) * sigma + 1e-6)
    )
    z = cdf_evals[..., -1] - cdf_evals[..., 0]
    target_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
    target_probs = (target_probs / (z.unsqueeze(-1) + 1e-6)).reshape(
        *inp.shape[:-1], num_atoms
    )
    return target_probs


class _DeterministicActor(nn.Module):
    """Deterministic exportable policy head."""

    def __init__(self, mean_net: nn.Module) -> None:
        super().__init__()
        self.mean_net = mean_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mean_net(obs))

    def __getitem__(self, idx: int) -> nn.Module:
        return self.mean_net[idx]


def _concat_obs(obs: TensorDict, obs_groups: list[str]) -> torch.Tensor:
    obs_list = [obs[obs_group] for obs_group in obs_groups]
    return torch.cat(obs_list, dim=-1)


class ReppoPolicy(nn.Module):
    """Actor-only policy wrapper used by the REPPO runner.

    The module keeps the deterministic export path in ``actor`` and a separate
    mean network for stochastic training.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        actor_min_std: float = 0.1,
        ent_start: float = 1.0,
        kl_start: float = 1.0,
        **kwargs: dict,
    ) -> None:
        super().__init__()
        if kwargs:
            print(
                "ReppoPolicy.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys())
            )

        self.obs_groups = obs_groups
        self.actor_obs_groups = obs_groups.get("actor", obs_groups.get("policy", []))
        self.critic_obs_groups = obs_groups.get("critic", self.actor_obs_groups)

        self.actor_obs_dim = self._get_obs_dim(obs, self.actor_obs_groups)
        self.critic_obs_dim = self._get_obs_dim(obs, self.critic_obs_groups)

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self.actor_obs_dim)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(self.critic_obs_dim)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        self.actor_mean = MLP(self.actor_obs_dim, num_actions, actor_hidden_dims, activation=activation)
        self.actor_mean.init_weights(1.0)
        self.actor = _DeterministicActor(self.actor_mean)

        self.noise_std_type = noise_std_type
        self.actor_min_std = actor_min_std
        if noise_std_type == "scalar":
            self.std_param = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std_param = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(
                f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'."
            )

        self.log_temp = nn.Parameter(torch.log(torch.tensor(ent_start)))
        self.log_lagrange = nn.Parameter(torch.log(torch.tensor(kl_start)))

        Normal.set_default_validate_args(False)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: list[str]) -> int:
        obs_dim = 0
        for obs_group in obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"ReppoPolicy only supports 1D observations, got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return obs_dim

    def get_actor_obs(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, TensorDict):
            return _concat_obs(obs, self.actor_obs_groups)
        return obs

    def get_critic_obs(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, TensorDict):
            return _concat_obs(obs, self.critic_obs_groups)
        return obs

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))  # type: ignore
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))  # type: ignore

    def _get_std(self, mean: torch.Tensor) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            std = self.std_param.expand_as(mean)
        else:
            std = torch.exp(self.log_std_param).expand_as(mean)
        return std + self.actor_min_std

    def build_distribution(self, obs: TensorDict | torch.Tensor) -> TransformedDistribution:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        mean = self.actor_mean(actor_obs)
        std = self._get_std(mean)
        return TransformedDistribution(Normal(mean, std), [TanhTransform(cache_size=1)])

    def sample_actions(
        self, obs: TensorDict | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        mean = self.actor_mean(actor_obs)
        std = self._get_std(mean)
        dist = TransformedDistribution(Normal(mean, std), [TanhTransform(cache_size=1)])
        actions = dist.rsample()
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = Normal(mean, std).entropy().sum(dim=-1)
        return actions, log_prob, entropy, mean

    def get_actions_log_prob(self, obs: TensorDict | torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        dist = self.build_distribution(obs)
        return dist.log_prob(actions).sum(dim=-1)

    def forward(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        """Deterministic inference path used by play/export utilities."""
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.actor(actor_obs)

    def act_inference(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        return self.forward(obs)

    @property
    def output_std(self) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            return self.std_param.detach().clone()
        return torch.exp(self.log_std_param.detach().clone()) + self.actor_min_std


class ReppoCritic(nn.Module):
    """Distributional critic used by REPPO."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        hidden_dims: tuple[int, ...] | list[int] | None = None,
        activation: str = "elu",
        num_atoms: int = 101,
        vmin: float = -5.0,
        vmax: float = 5.0,
        aux_loss_mult: float = 1.0,
        **kwargs: dict,
    ) -> None:
        super().__init__()
        if kwargs:
            print(
                "ReppoCritic.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys())
            )

        self.obs_groups = obs_groups
        self.critic_obs_groups = obs_groups.get("critic", obs_groups.get("policy", []))
        self.obs_dim = self._get_obs_dim(obs, self.critic_obs_groups)
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.vmin = vmin
        self.vmax = vmax
        self.aux_loss_mult = aux_loss_mult

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        critic_hidden_dims = hidden_dims if hidden_dims is not None else critic_hidden_dims
        feature_dim = critic_hidden_dims[0] if critic_hidden_dims else 256
        self.feature_module = MLP(
            self.obs_dim + num_actions,
            feature_dim,
            critic_hidden_dims,
            activation=activation,
        )
        self.critic_module = MLP(
            feature_dim,
            num_atoms,
            critic_hidden_dims,
            activation=activation,
        )
        self.pred_module = MLP(
            feature_dim,
            feature_dim,
            critic_hidden_dims,
            activation=activation,
        )

        self.register_buffer(
            "value_bins",
            torch.linspace(vmin, vmax, num_atoms, dtype=torch.float32),
        )
        self.zero_dist = nn.Parameter(hl_gauss(torch.zeros(1), self.vmin, self.vmax, self.num_atoms))

    def _get_obs_dim(self, obs: TensorDict, obs_groups: list[str]) -> int:
        obs_dim = 0
        for obs_group in obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"ReppoCritic only supports 1D observations, got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return obs_dim

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return _concat_obs(obs, self.critic_obs_groups)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))  # type: ignore

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = self.critic_obs_normalizer(obs)
        inp = torch.cat([obs, action], dim=-1)
        features = self.feature_module(inp)
        next_pred = self.pred_module(features)
        logits = self.critic_module(features) + 40.9 * self.zero_dist
        value_cats = torch.softmax(logits, dim=-1)
        value = value_cats @ self.value_bins
        return value, logits, next_pred, features
