# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict


@dataclass
class _ReplayBufferState:
    capacity: int
    storage: list[TensorDict]


class TensorDictReplayBuffer:
    """Ring buffer for batched TensorDict transitions.

    The buffer stores individual transitions, but accepts a leading batch dimension when adding data.
    """

    def __init__(self, capacity: int, device: torch.device | str | None = "cpu") -> None:
        if capacity <= 0:
            raise ValueError(f"Replay buffer capacity must be positive, got {capacity}.")

        self.capacity = int(capacity)
        self.device = torch.device(device) if device is not None else None
        self._storage: list[TensorDict | None] = [None] * self.capacity
        self._next_idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def clear(self) -> None:
        self._storage = [None] * self.capacity
        self._next_idx = 0
        self._size = 0

    def add(self, transition: TensorDict | dict) -> None:
        """Add a transition or a batch of transitions to the buffer."""
        item = transition if isinstance(transition, TensorDict) else TensorDict.from_dict(transition)
        items = self._split_batch(item)

        for entry in items:
            stored = entry.clone()
            if self.device is not None:
                stored = stored.to(self.device)
            self._storage[self._next_idx] = stored
            self._next_idx = (self._next_idx + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> TensorDict:
        """Sample a random batch of transitions."""
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}.")
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample batch_size={batch_size} from replay buffer with size {self._size}."
            )

        indices = torch.randint(self._size, (batch_size,), generator=generator)
        ordered = self._ordered_storage()
        batch = TensorDict.stack([ordered[int(idx)] for idx in indices], dim=0)
        if device is not None:
            batch = batch.to(device)
        return batch

    def state_dict(self) -> dict[str, object]:
        """Serialize the buffer in chronological order."""
        return {
            "capacity": self.capacity,
            "storage": [item.clone() for item in self._ordered_storage()],
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore the buffer from ``state_dict``."""
        capacity = int(state_dict["capacity"])  # type: ignore[index]
        if capacity != self.capacity:
            raise ValueError(
                f"Replay buffer capacity mismatch: checkpoint has {capacity}, buffer has {self.capacity}."
            )

        self.clear()
        for entry in state_dict.get("storage", []):  # type: ignore[assignment]
            self.add(entry)

    def _ordered_storage(self) -> list[TensorDict]:
        if self._size == 0:
            return []

        start = (self._next_idx - self._size) % self.capacity
        ordered: list[TensorDict] = []
        for offset in range(self._size):
            entry = self._storage[(start + offset) % self.capacity]
            if entry is None:
                raise RuntimeError("Replay buffer storage is corrupted: missing entry.")
            ordered.append(entry)
        return ordered

    def _split_batch(self, item: TensorDict) -> list[TensorDict]:
        if len(item.batch_size) == 0:
            return [item]
        if len(item.batch_size) != 1:
            raise ValueError(
                "Replay buffer expects items with at most one leading batch dimension. "
                f"Got batch_size={tuple(item.batch_size)}."
            )

        batch_size = int(item.batch_size[0])
        return [item[idx].clone() for idx in range(batch_size)]
