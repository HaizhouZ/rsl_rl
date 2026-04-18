"""Tanh-squashed Gaussian distribution for bounded continuous action spaces."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import TanhTransform


class TanhNormal(TransformedDistribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    has_rsample = True

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        validate_args: bool | None = None,
        action_lower_bound: Tensor | float | None = None,
        action_upper_bound: Tensor | float | None = None,
    ) -> None:
        self.loc = loc
        self.scale = scale
        self.action_bounds_lower = action_lower_bound
        self.action_bounds_upper = action_upper_bound
        if self.action_bounds_lower is not None and self.action_bounds_upper is not None:
            self._mean_offset = (self.action_bounds_upper + self.action_bounds_lower) / 2.0
            self._scale = (self.action_bounds_upper - self.action_bounds_lower) / 2.0
            if not isinstance(self._mean_offset, Tensor):
                self._mean_offset = torch.tensor(self._mean_offset, dtype=loc.dtype, device=loc.device)
            if not isinstance(self._scale, Tensor):
                self._scale = torch.tensor(self._scale, dtype=loc.dtype, device=loc.device)
        else:
            self._mean_offset = torch.tensor(0.0, dtype=loc.dtype, device=loc.device)
            self._scale = torch.tensor(1.0, dtype=loc.dtype, device=loc.device)
        base_dist = Normal(loc, scale, validate_args=validate_args)
        super().__init__(base_dist, TanhTransform(cache_size=1), validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return self._mean_offset + self._scale * torch.tanh(self.loc)

    @property
    def mode(self) -> Tensor:
        return self.mean

    @property
    def stddev(self) -> Tensor:
        return self.scale * (1 - torch.tanh(self.loc) ** 2)

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)

        value = (value - self._mean_offset) / self._scale
        eps = torch.finfo(value.dtype).eps
        value = value.clamp(-1.0 + eps, 1.0 - eps)
        pre_tanh = torch.atanh(value)
        log_prob = self.base_dist.log_prob(pre_tanh)
        log_det_jacobian = 2.0 * (math.log(2.0) - pre_tanh - nn.functional.softplus(-2.0 * pre_tanh))
        return log_prob - log_det_jacobian - torch.log(self._scale)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        return super().rsample(sample_shape) * self._scale + self._mean_offset

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        return super().sample(sample_shape) * self._scale + self._mean_offset

    def entropy(self) -> Tensor:
        return self.base_dist.entropy() - torch.log(self._scale)

    def expand(self, batch_shape: torch.Size, _instance=None) -> "TanhNormal":
        new = self._get_checked_instance(TanhNormal, _instance)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new._mean_offset = self._mean_offset.expand(batch_shape)
        new._scale = self._scale.expand(batch_shape)
        base_dist = Normal(new.loc, new.scale, validate_args=False)
        super(TanhNormal, new).__init__(base_dist, TanhTransform(cache_size=1), validate_args=False)
        new._validate_args = self._validate_args
        return new


def log_prob_from_tanh_normal(
    value: Tensor,
    loc: Tensor,
    scale: Tensor,
    pre_tanh_value: Tensor | None = None,
) -> Tensor:
    if pre_tanh_value is None:
        eps = torch.finfo(value.dtype).eps
        value = value.clamp(-1.0 + eps, 1.0 - eps)
        pre_tanh_value = torch.atanh(value)

    var = scale**2
    log_scale = torch.log(scale)
    log_prob = -0.5 * (((pre_tanh_value - loc) ** 2) / var + 2 * log_scale + math.log(2 * math.pi))
    log_det_jacobian = 2.0 * (math.log(2.0) - pre_tanh_value - nn.functional.softplus(-2.0 * pre_tanh_value))
    return log_prob - log_det_jacobian
