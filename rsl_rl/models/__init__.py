# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""

from .cnn_model import CNNModel
from .reppo_model import ReppoCritic, ReppoPolicy
from .mlp_model import MLPModel
from .rnn_model import RNNModel
from rsl_rl.modules import GaussianDistribution


class ActorCritic(MLPModel):
    """Compatibility wrapper for legacy PPO configs."""

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        hidden_dims=(256, 256, 256),
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        obs_normalization=False,
        distribution_cfg=None,
        **kwargs,
    ):
        if kwargs:
            # Ignore legacy or unsupported keys rather than failing old configs.
            kwargs.clear()
        if distribution_cfg is None and obs_set == "actor":
            distribution_cfg = {
                "class_name": "GaussianDistribution",
                "init_std": init_noise_std,
                "std_type": noise_std_type,
            }
        if obs_set == "actor":
            obs_normalization = actor_obs_normalization if actor_obs_normalization is not None else obs_normalization
        elif obs_set == "critic":
            obs_normalization = (
                critic_obs_normalization if critic_obs_normalization is not None else obs_normalization
            )
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

__all__ = [
    "CNNModel",
    "ActorCritic",
    "MLPModel",
    "ReppoCritic",
    "ReppoPolicy",
    "RNNModel",
]
