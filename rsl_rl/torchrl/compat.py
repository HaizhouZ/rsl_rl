# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TorchRL compatibility helpers.

This module provides a lightweight compatibility layer that adapts the existing
:class:`rsl_rl.env.VecEnv` contract to a TorchRL-like TensorDict API without
changing training internals.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv


class TorchRLVecEnvWrapper:
    """Adapt an :class:`~rsl_rl.env.VecEnv` to TorchRL-style TensorDict I/O.

    The wrapper does **not** depend on ``torchrl`` at runtime and is intended as
    an integration shim. It returns TensorDict objects with keys frequently used
    in TorchRL pipelines:

    - ``"observation"``: concatenated observation tensor.
    - ``"observations"``: original observation TensorDict from the environment.
    - ``"reward"``: reward tensor shaped ``[num_envs, 1]``.
    - ``"done"``, ``"terminated"``, ``"truncated"``: boolean flags.
    - ``"next"``: next-step TensorDict payload in step outputs.
    """

    def __init__(self, env: VecEnv, observation_keys: list[str] | None = None) -> None:
        """Initialize the wrapper.

        Args:
            env: Wrapped vectorized environment.
            observation_keys: Optional ordered observation group keys to
                concatenate into the ``"observation"`` tensor. If ``None``, all
                keys from the current observation TensorDict are used.
        """
        self.env = env
        self.observation_keys = observation_keys

    def reset(self) -> TensorDict:
        """Return a TorchRL-style initial TensorDict from ``env.get_observations()``."""
        obs = self.env.get_observations()
        return self._build_reset_tensordict(obs)

    def step(self, action: torch.Tensor | TensorDict) -> TensorDict:
        """Step the wrapped environment using tensor or TensorDict action input.

        Args:
            action: Either a raw action tensor shaped ``[num_envs, num_actions]``
                or a TensorDict containing an ``"action"`` key.
        """
        actions = self.extract_action(action)
        obs, rewards, dones, extras = self.env.step(actions)
        return self._build_step_tensordict(obs, rewards, dones, extras)

    def extract_action(self, action: torch.Tensor | TensorDict) -> torch.Tensor:
        """Extract action tensor from either a tensor or TensorDict payload."""
        if isinstance(action, TensorDict):
            if "action" not in action.keys(include_nested=False):
                raise KeyError("Action TensorDict must contain an 'action' key.")
            return action["action"]
        return action

    def _build_reset_tensordict(self, obs: TensorDict) -> TensorDict:
        done = torch.zeros(self.env.num_envs, 1, dtype=torch.bool, device=self.env.device)
        td = TensorDict({}, batch_size=[self.env.num_envs], device=self.env.device)
        td["observation"] = self._concat_observations(obs)
        td["observations"] = obs.clone()
        td["done"] = done
        td["terminated"] = done.clone()
        td["truncated"] = done.clone()
        return td

    def _build_step_tensordict(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
    ) -> TensorDict:
        done = dones.view(-1, 1).to(dtype=torch.bool)
        time_outs = extras.get("time_outs")
        if time_outs is None:
            truncated = torch.zeros_like(done)
        else:
            truncated = time_outs.view(-1, 1).to(dtype=torch.bool)
        terminated = done & ~truncated

        next_td = TensorDict({}, batch_size=[self.env.num_envs], device=self.env.device)
        next_td["observation"] = self._concat_observations(obs)
        next_td["observations"] = obs.clone()
        next_td["reward"] = rewards.view(-1, 1)
        next_td["done"] = done
        next_td["terminated"] = terminated
        next_td["truncated"] = truncated

        if "log" in extras:
            next_td["log"] = extras["log"]

        out = TensorDict({}, batch_size=[self.env.num_envs], device=self.env.device)
        out["next"] = next_td
        return out

    def _concat_observations(self, obs: TensorDict) -> torch.Tensor:
        if self.observation_keys is None:
            keys = list(obs.keys(include_nested=False))
        else:
            keys = self.observation_keys
        return torch.cat([obs[key] for key in keys], dim=-1)


def to_torchrl_action_tensordict(actions: torch.Tensor) -> TensorDict:
    """Wrap a raw action tensor in a TorchRL-style TensorDict."""
    return TensorDict({"action": actions}, batch_size=[actions.shape[0]], device=actions.device)
