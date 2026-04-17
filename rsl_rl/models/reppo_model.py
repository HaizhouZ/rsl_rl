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

from rsl_rl.modules import EmpiricalNormalization


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


class _ActorMeanHead(nn.Module):
    def __init__(self, actor_model: nn.Module, num_actions: int) -> None:
        super().__init__()
        self.actor_model = actor_model
        self.num_actions = num_actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = torch.split(self.actor_model(obs), self.num_actions, dim=-1)
        return mean

    def __getitem__(self, idx: int) -> nn.Module:
        if hasattr(self.actor_model, "net"):
            return self.actor_model.net[idx]  # type: ignore[index]
        raise TypeError("Underlying actor model does not support indexing")


def _resolve_hidden_dims(
    explicit_dims: tuple[int, ...] | list[int] | None,
    fallback_hidden_dim: int,
    fallback_layers: int,
) -> tuple[int, ...]:
    if explicit_dims is not None and len(explicit_dims) > 0:
        return tuple(explicit_dims)
    return tuple([fallback_hidden_dim] * fallback_layers)


def _activation(name: str | None) -> nn.Module:
    if name in ("swish", "silu"):
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name is None:
        return nn.Identity()
    raise ValueError(f"Unsupported REPPO activation: {name}")


class _FCNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "swish",
        use_norm: bool = True,
        input_activation: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        if input_activation:
            layers.append(_activation(activation))
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_norm:
                layers.append(nn.RMSNorm(hidden_dim))
            layers.append(_activation(activation))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
        actor_hidden_dims: tuple[int, ...] | list[int] = (512, 512, 512),
        critic_hidden_dims: tuple[int, ...] | list[int] = (512, 512, 512),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        actor_min_std: float = 0.0,
        ent_start: float = 0.01,
        kl_start: float = 0.01,
        actor_hidden_dim: int = 512,
        num_actor_layers: int = 3,
        use_actor_norm: bool = True,
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

        actor_hidden_dims = _resolve_hidden_dims(actor_hidden_dims, actor_hidden_dim, num_actor_layers)
        self.actor_model = _FCNN(
            self.actor_obs_dim,
            2 * num_actions,
            actor_hidden_dims,
            activation=activation if activation is not None else "swish",
            use_norm=use_actor_norm,
        )
        self.actor_mean = _ActorMeanHead(self.actor_model, num_actions)
        self.actor = _DeterministicActor(self.actor_mean)

        self.actor_min_std = actor_min_std
        self.init_noise_std = init_noise_std
        self.register_buffer("_cached_output_std", init_noise_std * torch.ones(num_actions))

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

    def normalize_actor_obs(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        return self.actor_obs_normalizer(self.get_actor_obs(obs))

    def normalize_critic_obs(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        return self.critic_obs_normalizer(self.get_critic_obs(obs))

    @property
    def num_actions(self) -> int:
        return self._cached_output_std.shape[-1]

    def _distribution_from_normalized(
        self, normalized_obs: torch.Tensor
    ) -> tuple[TransformedDistribution, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.actor_model(normalized_obs)
        mean, log_std = torch.split(out, out.shape[-1] // 2, dim=-1)
        std = torch.exp(log_std) + self.actor_min_std
        self._cached_output_std.copy_(std.mean(dim=0).detach())
        dist = TransformedDistribution(Normal(mean, std), [TanhTransform(cache_size=1)])
        return dist, torch.tanh(mean), torch.exp(self.log_temp), torch.exp(self.log_lagrange)

    def build_distribution(self, obs: TensorDict | torch.Tensor) -> TransformedDistribution:
        dist, _, _, _ = self._distribution_from_normalized(self.normalize_actor_obs(obs))
        return dist

    def build_distribution_from_normalized(self, normalized_obs: torch.Tensor) -> TransformedDistribution:
        dist, _, _, _ = self._distribution_from_normalized(normalized_obs)
        return dist

    def sample_actions(
        self, obs: TensorDict | torch.Tensor, normalized: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_obs = obs if normalized else self.normalize_actor_obs(obs)
        dist, deterministic_actions, _, _ = self._distribution_from_normalized(actor_obs)
        actions = dist.rsample()
        clipped_actions = actions.clamp(-1 + 1e-6, 1 - 1e-6)
        log_prob = dist.log_prob(clipped_actions).sum(dim=-1)
        entropy = -log_prob
        return actions, log_prob, entropy, deterministic_actions

    def sample_actions_from_normalized(
        self, normalized_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, deterministic_actions, temperature, beta = self._distribution_from_normalized(normalized_obs)
        actions = dist.rsample()
        clipped_actions = actions.clamp(-1 + 1e-6, 1 - 1e-6)
        log_prob = dist.log_prob(clipped_actions).sum(dim=-1)
        entropy = -log_prob
        return actions, log_prob, entropy, deterministic_actions, temperature, beta

    def get_actions_log_prob(self, obs: TensorDict | torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        dist = self.build_distribution(obs)
        return dist.log_prob(actions).sum(dim=-1)

    def forward(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        """Deterministic inference path used by play/export utilities."""
        return self.actor(self.normalize_actor_obs(obs))

    def act_inference(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        return self.forward(obs)

    @property
    def output_std(self) -> torch.Tensor:
        return self._cached_output_std.detach().clone()


class ReppoCritic(nn.Module):
    """Distributional critic used by REPPO."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int, ...] | list[int] = (512, 512, 512),
        critic_hidden_dims: tuple[int, ...] | list[int] = (512, 512, 512),
        hidden_dims: tuple[int, ...] | list[int] | None = None,
        activation: str = "elu",
        num_atoms: int = 151,
        vmin: float = 0.0,
        vmax: float = 150.0,
        aux_loss_mult: float = 0.0,
        critic_hidden_dim: int = 512,
        use_critic_norm: bool = True,
        use_encoder_norm: bool = False,
        num_critic_encoder_layers: int = 2,
        num_critic_head_layers: int = 2,
        num_critic_pred_layers: int = 2,
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
        critic_hidden_dims = _resolve_hidden_dims(critic_hidden_dims, critic_hidden_dim, num_critic_encoder_layers)
        feature_dim = critic_hidden_dims[-1] if critic_hidden_dims else critic_hidden_dim
        head_hidden_dims = tuple([feature_dim] * max(0, num_critic_head_layers - 1))
        pred_hidden_dims = tuple([feature_dim] * max(0, num_critic_pred_layers - 1))
        self.feature_module = _FCNN(
            self.obs_dim + num_actions,
            feature_dim,
            critic_hidden_dims,
            activation=activation if activation is not None else "swish",
            use_norm=use_critic_norm,
        )
        self.critic_module = _FCNN(
            feature_dim,
            num_atoms,
            head_hidden_dims,
            activation=activation if activation is not None else "swish",
            use_norm=use_critic_norm,
            input_activation=True,
        )
        self.pred_module = _FCNN(
            feature_dim,
            feature_dim,
            pred_hidden_dims,
            activation=activation if activation is not None else "swish",
            use_norm=use_critic_norm,
            input_activation=True,
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

    def normalize_critic_obs(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        return self.critic_obs_normalizer(self.get_critic_obs(obs) if isinstance(obs, TensorDict) else obs)

    def forward_normalized(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inp = torch.cat([obs, action], dim=-1)
        features = self.feature_module(inp)
        next_pred = self.pred_module(features)
        logits = self.critic_module(features) + 40.9 * self.zero_dist
        value_cats = torch.softmax(logits, dim=-1)
        value = value_cats @ self.value_bins
        return value, logits, next_pred, features

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_normalized(self.normalize_critic_obs(obs), action)
