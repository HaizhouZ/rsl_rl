from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.storage import TensorDictReplayBuffer


def _make_transition(step: int, batch_size: int | None = None) -> TensorDict:
    if batch_size is None:
        obs = torch.tensor([step, step + 1], dtype=torch.float32)
        critic_obs = torch.tensor([step + 10, step + 11], dtype=torch.float32)
        next_obs = torch.tensor([step + 20, step + 21], dtype=torch.float32)
        reward = torch.tensor([float(step)], dtype=torch.float32)
        done = torch.tensor([float(step % 2)], dtype=torch.float32)
        action = torch.tensor([step + 30], dtype=torch.float32)
        return TensorDict(
            {
                "obs": obs,
                "critic_obs": critic_obs,
                "actions": action,
                "next": TensorDict(
                    {
                        "obs": next_obs,
                        "rewards": reward,
                        "dones": done,
                    },
                    batch_size=(),
                ),
            },
            batch_size=(),
        )

    obs = torch.stack(
        [torch.tensor([step + i, step + i + 1], dtype=torch.float32) for i in range(batch_size)],
        dim=0,
    )
    critic_obs = torch.stack(
        [torch.tensor([step + 10 + i, step + 11 + i], dtype=torch.float32) for i in range(batch_size)],
        dim=0,
    )
    next_obs = torch.stack(
        [torch.tensor([step + 20 + i, step + 21 + i], dtype=torch.float32) for i in range(batch_size)],
        dim=0,
    )
    reward = torch.arange(step, step + batch_size, dtype=torch.float32).unsqueeze(-1)
    done = (torch.arange(batch_size) % 2).to(torch.float32).unsqueeze(-1)
    action = torch.arange(step + 30, step + 30 + batch_size, dtype=torch.float32).unsqueeze(-1)
    return TensorDict(
        {
            "obs": obs,
            "critic_obs": critic_obs,
            "actions": action,
            "next": TensorDict(
                {
                    "obs": next_obs,
                    "rewards": reward,
                    "dones": done,
                },
                batch_size=(batch_size,),
            ),
        },
        batch_size=(batch_size,),
    )


def test_replay_buffer_add_sample_and_roundtrip() -> None:
    buffer = TensorDictReplayBuffer(capacity=4)

    buffer.add(_make_transition(0, batch_size=2))
    buffer.add(_make_transition(1))

    assert len(buffer) == 3

    sample = buffer.sample(2, generator=torch.Generator().manual_seed(0))
    assert sample.batch_size == torch.Size([2])
    assert sample["obs"].shape == (2, 2)
    assert sample["next"]["obs"].shape == (2, 2)

    state = buffer.state_dict()
    assert len(state["storage"]) == 3
    assert torch.equal(state["storage"][0]["obs"], torch.tensor([0.0, 1.0]))
    assert torch.equal(state["storage"][1]["obs"], torch.tensor([1.0, 2.0]))
    assert torch.equal(state["storage"][2]["obs"], torch.tensor([1.0, 2.0]))

    restored = TensorDictReplayBuffer(capacity=4)
    restored.load_state_dict(state)
    assert len(restored) == 3
    assert torch.equal(restored.state_dict()["storage"][0]["critic_obs"], torch.tensor([10.0, 11.0]))


def test_replay_buffer_overwrites_old_items() -> None:
    buffer = TensorDictReplayBuffer(capacity=2)

    buffer.add(_make_transition(0))
    buffer.add(_make_transition(1))
    buffer.add(_make_transition(2))

    state = buffer.state_dict()
    assert len(state["storage"]) == 2
    assert torch.equal(state["storage"][0]["obs"], torch.tensor([1.0, 2.0]))
    assert torch.equal(state["storage"][1]["obs"], torch.tensor([2.0, 3.0]))
