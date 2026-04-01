# agent.md

## Purpose
This repository (`rsl_rl`, package name `rsl-rl-lib`) is a lightweight, GPU-first reinforcement learning library focused on robot learning workflows. It provides:
- On-policy RL training (PPO).
- Student-teacher behavior distillation.
- Optional extensions (RND intrinsic reward and symmetry augmentation/loss).
- Modular policy/model blocks (MLP/CNN/RNN + configurable action distributions).

## Repository Map
- `rsl_rl/algorithms/`: training algorithms (`PPO`, `Distillation`).
- `rsl_rl/runners/`: orchestration loops (`OnPolicyRunner`, `DistillationRunner`).
- `rsl_rl/models/`: high-level policy/value model wrappers (`MLPModel`, `CNNModel`, `RNNModel`).
- `rsl_rl/modules/`: neural network primitives and distributions (`MLP`, `CNN`, `RNN`, `GaussianDistribution`, normalization modules).
- `rsl_rl/storage/`: rollout buffer and batch generators (`RolloutStorage`).
- `rsl_rl/env/`: vectorized environment interface (`VecEnv`) expected by runners/algorithms.
- `rsl_rl/extensions/`: optional training extensions (RND and symmetry config resolution).
- `rsl_rl/utils/`: utility resolvers and helpers (`resolve_callable`, `resolve_obs_groups`, etc.).
- `tests/`: unit/integration tests.

## Core Runtime Interfaces

### 1) Environment contract: `VecEnv`
Any environment must implement:
- `get_observations() -> TensorDict`
- `step(actions: Tensor) -> (obs: TensorDict, rewards: Tensor, dones: Tensor, extras: dict)`

Required attributes include:
- `num_envs`, `num_actions`
- `max_episode_length`, `episode_length_buf`
- `device`, `cfg`

Important `extras` keys consumed by training code:
- `"time_outs"`: used for timeout bootstrapping in PPO.
- `"log"`: additional scalar/tensor logging payloads.

### 2) Runner contract
`OnPolicyRunner` handles:
- Multi-GPU bootstrap from `WORLD_SIZE/LOCAL_RANK/RANK`.
- Algorithm construction from config via `resolve_callable`.
- Rollout collection + update loop.
- Logging, checkpoint save/load, and export to JIT/ONNX.

`DistillationRunner` extends `OnPolicyRunner` and enforces that teacher weights are loaded before learning.

### 3) Algorithm contract
Both algorithm classes expose a similar lifecycle used by runners:
- `act(obs)`
- `process_env_step(obs, rewards, dones, extras)`
- `compute_returns(obs)`
- `update() -> dict[str, float]`
- `train_mode()`, `eval_mode()`
- `save()`, `load(...)`
- `get_policy()`

`PPO` specifics:
- Uses actor/critic models + rollout storage.
- Supports adaptive KL learning-rate schedule.
- Supports optional RND and symmetry logic.

`Distillation` specifics:
- Uses student (trainable) and teacher (target) policies.
- Optimizes behavior loss (`mse` or `huber`).
- Does not use return computation.

### 4) Storage contract: `RolloutStorage`
`RolloutStorage` is shared by both RL and distillation:
- `Transition`: per-step record container.
- `Batch`: yielded training mini-batch view.
- `add_transition(...)`, `clear()`
- Distillation iterator: `generator()`
- PPO iterators:
  - `mini_batch_generator(...)` for feedforward models
  - `recurrent_mini_batch_generator(...)` for recurrent models

## Model and Module Layering
- `models/*_model.py` are policy/value wrappers that:
  - select configured observation groups,
  - optionally normalize observations,
  - route features through backbone module,
  - optionally attach stochastic distributions.
- `modules/*` provide reusable blocks:
  - feature extractors (`MLP`, `CNN`, `RNN`),
  - stochastic output distributions,
  - normalization primitives.

## Configuration & Resolution Patterns
The codebase heavily uses string-to-callable resolution and observation mapping:
- `resolve_callable(...)`: accepts direct callable or import path string.
- `resolve_obs_groups(...)`: validates/fills observation set mappings (e.g., actor/critic/student/teacher).
- `resolve_optimizer(...)`, `resolve_nn_activation(...)`: map string names to torch components.

This allows external projects to inject custom classes/functions through config without modifying rsl_rl internals.

## Typical Training Flow (On-policy PPO)
1. Create a `VecEnv` implementation.
2. Build runner with `train_cfg`.
3. Runner resolves and constructs algorithm/models/storage.
4. Iterative loop:
   - `act` -> `env.step` -> `process_env_step` (for `num_steps_per_env`)
   - `compute_returns`
   - `update`
5. Log metrics and periodically checkpoint.
6. Optionally export policy to TorchScript or ONNX.

## Development Notes
- Python requirement: `>=3.9`.
- Key dependencies: PyTorch, TensorDict, NumPy, ONNX stack.
- Style/contrib expectations: PEP 8, Google-style docstrings, run `pre-commit run --all-files`.

## Fast Start Commands
```bash
pip install -e .
pre-commit run --all-files
```
