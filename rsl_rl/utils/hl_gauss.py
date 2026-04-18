"""HL-Gauss transform for distributional value prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class HLGaussTransform:
    def __init__(
        self,
        min_value: float,
        max_value: float,
        num_bins: int,
        sigma_ratio: float = 0.75,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.bin_width = (max_value - min_value) / (num_bins - 1)
        self.sigma = self.bin_width * sigma_ratio
        self.bin_edges = torch.linspace(
            min_value - self.bin_width / 2,
            max_value + self.bin_width / 2,
            num_bins + 1,
        )
        self.support = torch.linspace(min_value, max_value, num_bins)

    def to(self, device: torch.device) -> "HLGaussTransform":
        self.bin_edges = self.bin_edges.to(device)
        self.support = self.support.to(device)
        return self

    def embed_targets(self, targets: Tensor) -> Tensor:
        targets = targets.clamp(self.min_value, self.max_value)
        bin_edges = self.bin_edges.to(targets.device)
        cdf_evals = torch.erf((bin_edges - targets.unsqueeze(-1)) / (self.sigma * (2**0.5)))
        target_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        z = cdf_evals[..., -1:] - cdf_evals[..., :1]
        target_probs = target_probs / z
        return target_probs

    def decode(self, logits: Tensor) -> Tensor:
        support = self.support.to(logits.device)
        probs = torch.softmax(logits, dim=-1)
        return (probs * support).sum(dim=-1)


class HLGaussLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        min_value: float,
        max_value: float,
        num_bins: int,
        sigma: float = 0.75,
        offset_mult: float = 0.0,
    ):
        super().__init__()
        self.transform = HLGaussTransform(min_value, max_value, num_bins, sigma)
        self.linear = nn.Linear(in_features, num_bins)
        self.offset = nn.Parameter(self.embed_targets(torch.tensor([0.0])) * offset_mult, requires_grad=True)

    @property
    def num_bins(self) -> int:
        return self.transform.num_bins

    @property
    def support(self) -> Tensor:
        return self.transform.support

    def forward(self, x: Tensor, return_logits: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        logits = self.linear(x) + self.offset
        values = self.transform.decode(logits)
        if return_logits:
            return values, logits
        return values

    def embed_targets(self, targets: Tensor) -> Tensor:
        return self.transform.embed_targets(targets)

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        soft_targets = self.embed_targets(targets)
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()


def embed_targets(
    targets: Tensor,
    min_value: float,
    max_value: float,
    num_bins: int,
    sigma_ratio: float = 0.75,
) -> Tensor:
    transform = HLGaussTransform(min_value, max_value, num_bins, sigma_ratio)
    return transform.embed_targets(targets)
