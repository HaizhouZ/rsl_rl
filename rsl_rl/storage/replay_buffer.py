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


@dataclass
class _BatchedReplayBufferState:
    capacity: int
    num_envs: int
    storage: list[list[TensorDict]]


class TensorDictReplayBuffer:
    """Ring buffer for batched TensorDict transitions.

    The buffer stores individual transitions, but accepts a leading batch dimension when adding data.
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device | str | None = "cpu",
        num_envs: int | None = None,
        n_steps: int = 1,
        gamma: float = 0.99,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"Replay buffer capacity must be positive, got {capacity}.")

        self.capacity = int(capacity)
        self.device = torch.device(device) if device is not None else None
        self.num_envs = int(num_envs) if num_envs is not None else None
        self.n_steps = max(1, int(n_steps))
        self.gamma = float(gamma)
        self._batched = self.num_envs is not None and self.n_steps > 1
        if self._batched:
            self._storage: list[list[TensorDict | None]] = [
                [None] * self.capacity for _ in range(self.num_envs or 0)
            ]
        else:
            self._storage: list[TensorDict | None] = [None] * self.capacity
        self._next_idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def clear(self) -> None:
        if self._batched:
            self._storage = [[None] * self.capacity for _ in range(self.num_envs or 0)]
        else:
            self._storage = [None] * self.capacity
        self._next_idx = 0
        self._size = 0

    def add(self, transition: TensorDict | dict) -> None:
        """Add a transition or a batch of transitions to the buffer."""
        item = transition if isinstance(transition, TensorDict) else TensorDict.from_dict(transition)
        items = self._split_batch(item)

        if self._batched:
            if self.num_envs is None:
                raise RuntimeError("Batched replay buffer is missing num_envs.")
            if len(items) != self.num_envs:
                raise ValueError(
                    f"Batched replay buffer expects batch_size={self.num_envs}, got {len(items)}."
                )
            for env_idx, entry in enumerate(items):
                stored = entry.clone()
                if self.device is not None:
                    stored = stored.to(self.device)
                self._storage[env_idx][self._next_idx] = stored
            self._next_idx = (self._next_idx + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
            return

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

        if self._batched:
            return self._sample_batched(batch_size, device=device, generator=generator)

        indices = torch.randint(self._size, (batch_size,), generator=generator)
        ordered = self._ordered_storage()
        batch = TensorDict.stack([ordered[int(idx)] for idx in indices], dim=0)
        if device is not None:
            batch = batch.to(device)
        return batch

    def state_dict(self) -> dict[str, object]:
        """Serialize the buffer in chronological order."""
        if self._batched:
            return {
                "capacity": self.capacity,
                "num_envs": self.num_envs,
                "n_steps": self.n_steps,
                "gamma": self.gamma,
                "storage": [[item.clone() for item in self._ordered_storage_for_env(env_idx)] for env_idx in range(self.num_envs or 0)],
            }
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
        storage = state_dict.get("storage", [])
        if self._batched:
            if not isinstance(storage, list):
                raise TypeError("Batched replay buffer state must contain a list of per-env storages.")
            if len(storage) != (self.num_envs or 0):
                raise ValueError(
                    f"Batched replay buffer state has {len(storage)} envs, expected {self.num_envs}."
                )
            if len(storage) == 0:
                return
            size = len(storage[0])
            for env_storage in storage:
                if len(env_storage) != size:
                    raise ValueError("Batched replay buffer env storages must have equal length.")
            self._size = min(size, self.capacity)
            self._next_idx = self._size % self.capacity
            for env_idx, env_storage in enumerate(storage):
                for slot, entry in enumerate(env_storage[: self.capacity]):
                    self._storage[env_idx][slot] = entry.clone()
            return

        for entry in storage:  # type: ignore[assignment]
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

    def _ordered_storage_for_env(self, env_idx: int) -> list[TensorDict]:
        if not self._batched:
            raise RuntimeError("Batched storage requested from a flat replay buffer.")
        if self._size == 0:
            return []

        start = (self._next_idx - self._size) % self.capacity
        ordered: list[TensorDict] = []
        for offset in range(self._size):
            entry = self._storage[env_idx][(start + offset) % self.capacity]
            if entry is None:
                raise RuntimeError("Replay buffer storage is corrupted: missing entry.")
            ordered.append(entry)
        return ordered

    def _sample_batched(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> TensorDict:
        if self.num_envs is None:
            raise RuntimeError("Batched replay buffer is missing num_envs.")
        if self.n_steps <= 1:
            return self._sample_batched_single_step(batch_size, device=device, generator=generator)
        if self._size < self.n_steps:
            raise RuntimeError(
                f"Cannot sample n-step batch with size {self._size} and n_steps={self.n_steps}."
            )

        valid_starts = self._size - self.n_steps + 1
        env_indices = torch.randint(self.num_envs, (batch_size,), generator=generator)
        start_indices = torch.randint(valid_starts, (batch_size,), generator=generator)
        samples = [
            self._sample_sequence(int(env_idx), int(start_idx))
            for env_idx, start_idx in zip(env_indices.tolist(), start_indices.tolist(), strict=False)
        ]
        batch = TensorDict.stack(samples, dim=0)
        if device is not None:
            batch = batch.to(device)
        return batch

    def _sample_batched_single_step(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> TensorDict:
        if self.num_envs is None:
            raise RuntimeError("Batched replay buffer is missing num_envs.")
        env_indices = torch.randint(self.num_envs, (batch_size,), generator=generator)
        start_indices = torch.randint(self._size, (batch_size,), generator=generator)
        samples = [
            self._ordered_storage_for_env(int(env_idx))[int(start_idx)]
            for env_idx, start_idx in zip(env_indices.tolist(), start_indices.tolist(), strict=False)
        ]
        batch = TensorDict.stack(samples, dim=0)
        if device is not None:
            batch = batch.to(device)
        return batch

    def _sample_sequence(self, env_idx: int, start_idx: int) -> TensorDict:
        ordered = self._ordered_storage_for_env(env_idx)
        sequence = ordered[start_idx : start_idx + self.n_steps]
        if not sequence:
            raise RuntimeError("Empty n-step sequence sampled from replay buffer.")

        first = sequence[0]
        reward = torch.zeros_like(first["next"]["rewards"])
        discount = 1.0
        effective_n_steps = 0
        final_next = first["next"]
        for entry in sequence:
            next_td = entry["next"]
            reward = reward + discount * next_td["rewards"]
            effective_n_steps += 1
            final_next = next_td
            done = bool(next_td["dones"].reshape(-1)[0].item())
            truncation = bool(next_td.get("truncations", torch.zeros_like(next_td["dones"])).reshape(-1)[0].item())
            if done or truncation:
                break
            discount *= self.gamma

        return TensorDict(
            {
                "observations": first["observations"].clone(),
                "actions": first["actions"].clone(),
                "next": TensorDict(
                    {
                        "observations": final_next["observations"].clone(),
                        "rewards": reward.clone(),
                        "dones": final_next["dones"].clone(),
                        "truncations": final_next.get(
                            "truncations", torch.zeros_like(final_next["dones"])
                        ).clone(),
                        "effective_n_steps": torch.tensor(
                            [float(effective_n_steps)],
                            device=reward.device,
                            dtype=torch.float32,
                        ),
                    },
                    batch_size=(),
                ),
            },
            batch_size=(),
        )

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
