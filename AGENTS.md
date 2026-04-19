# AGENTS.md

## Scope

`rsl_rl` is the core RL library. In this workspace it should stay REPPO-focused:
- keep the `Reppo` implementation in `rsl_rl`,
- keep legacy PPO and distillation behavior stable,
- and keep any `mjlab2`-specific compatibility logic out of this repository.

If a change is only needed because of `mjlab2` integration, implement the adapter on the `mjlab2` side.

## Current Boundary

The current public additions are:
- `rsl_rl.modules.ActorQ`
- `rsl_rl.algorithms.REPPO`
- `rsl_rl.runners.ReppoRunner`

The existing `OnPolicyRunner`, PPO path, logger behavior, and checkpoint format should not be broadened for `mjlab2` convenience unless there is a direct `rsl_rl` requirement.

## Working Rules

1. Prefer small, local changes.
2. Keep exports in `rsl_rl/models/__init__.py` and `rsl_rl/runners/__init__.py` aligned with the actual public surface.
3. Treat changes to `PPO`, `OnPolicyRunner`, and logger utilities as high risk.
4. Add adapter logic in `mjlab2` rather than teaching `rsl_rl` about old `mjlab2` assumptions.
5. Preserve the ability to run the full `mjlab2` suite against this checkout.

## Validation

When touching REPPO code, check at least:
- `python -m py_compile` for the edited files,
- a REPPO smoke test in `rsl_rl`,
- and `mjlab2`'s pytest suite when the public runner or checkpoint surface changes.

## Non-Goals

- Do not vendor the reference `reppo` repository.
- Do not reintroduce `mjlab2` compatibility shims into `rsl_rl`.
- Do not rewrite unrelated PPO or distillation code while iterating on REPPO.

## FastTD3 Port Checklist

### Baseline audit
- [x] Review the reference FastTD3 entrypoint, replay buffer, and shared network code.
- [x] Confirm `rsl_rl` has no off-policy runner or replay-buffer infrastructure today.
- [x] Decide to add FastTD3 as a separate off-policy stack instead of forcing it through `OnPolicyRunner`.

### `rsl_rl` implementation
- [x] Add a TensorDict replay buffer under `rsl_rl/storage/`.
- [x] Add a FastTD3 algorithm module under `rsl_rl/algorithms/`.
- [x] Add an off-policy runner under `rsl_rl/runners/`.
- [x] Add actor/critic wrappers if the existing `MLPModel` is not sufficient.
- [x] Export the new public classes from package `__init__` files.
- [x] Add FastTD3-specific config handling.
- [x] Keep FastTD3 checkpoint loading strict to native state_dict keys.
- [x] Bring FastTD3 behavior closer to the reference distributional critic / actor-noise path.

### `mjlab2` integration
- [x] Add one task config that selects the FastTD3 runner.
- [x] Keep any legacy config or checkpoint translation in `mjlab2`.
- [ ] Avoid touching the main `mjlab2` training script unless a hard compatibility gap appears.

### Validation
- [x] Add a replay-buffer unit test.
- [x] Add a single-step FastTD3 smoke test.
- [x] Run the `mjlab2` suite against the local `rsl_rl` checkout after the first end-to-end slice.

## Progress Log

- 2026-04-01: Audit complete. No off-policy infrastructure exists in `rsl_rl`; FastTD3 needs replay-buffer and target-network support.
- 2026-04-01: Starting with replay-buffer infrastructure in `rsl_rl/storage/`.
- 2026-04-01: Replay buffer implemented in `rsl_rl/storage/replay_buffer.py` and covered by `tests/storage/test_replay_buffer.py`.
- 2026-04-01: Added FastTD3 actor/critic wrappers, algorithm scaffold, and off-policy runner under `rsl_rl`.
- 2026-04-01: FastTD3 smoke test passes on the dummy env path. Remaining work is config wiring and parity tightening.
- 2026-04-01: FastTD3 config handling now requires native checkpoint keys and rejects legacy fallback shapes.
- 2026-04-01: FastTD3 updated to use distributional critics, reference-style exploration noise, and clipped-double-Q actor updates.
- 2026-04-01: Parity pass aligned the FastTD3 actor/critic defaults with the reference architecture and added optional reward normalization support.
- 2026-04-01: FastTD3 now supports n-step replay sampling and cosine LR schedules, and REPPO defaults were aligned with the reference TorchRL config.
- 2026-04-01: Replay sampling was tightened to an env-aware trajectory path, matching the reference minibatch geometry more closely.
- 2026-04-01: FastTD3 now waits on env-step history for learning and uses a per-env minibatch sample contract like the reference implementation.
