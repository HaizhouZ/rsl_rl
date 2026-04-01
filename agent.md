# agent.md

## Goal
Port the Torch `reppo` implementation into `rsl_rl` so `mjlab2` can use it through the existing `rsl_rl`-style training interfaces.

The target is not a literal copy of the reference repository. The target is a clean `rsl_rl` implementation that:
- fits the `VecEnv` contract used by `mjlab2`,
- preserves the current runner lifecycle where possible,
- keeps the `mjlab2` changes small,
- and avoids pulling the reference repo into `mjlab2` as a dependency.

## Port Strategy

### 1) Keep `mjlab2` on the existing runner boundary
`mjlab2` already wraps its environments with `RslRlVecEnvWrapper` and launches training through task-registered runner classes.
The migration should therefore prefer:
- a new `rsl_rl` algorithm implementation,
- reusable `rsl_rl` policy/value modules,
- and the existing `OnPolicyRunner` flow if the new algorithm can satisfy that contract.

This avoids rewriting the `mjlab2` training script.

### 2) Implement `Reppo` as a first-class `rsl_rl` algorithm
The `Reppo` algorithm should expose the same surface that `OnPolicyRunner` expects:
- `construct_algorithm(...)`
- `act(...)`
- `process_env_step(...)`
- `compute_returns(...)`
- `update()`
- `train_mode()` / `eval_mode()`
- `save()` / `load(...)`
- `get_policy()`

The internal logic can differ from PPO, but the outer contract should remain compatible.

### 3) Add dedicated model code rather than overloading PPO models
`reppo` needs:
- a squashed Gaussian actor,
- a distributional critic,
- empirical normalization,
- and export-friendly policy access for `mjlab2` task-specific ONNX exporters.

These should live in `rsl_rl` as reusable modules instead of being embedded inside `mjlab2`.

### 4) Keep `mjlab2` changes minimal
The expected `mjlab2` changes should be limited to:
- selecting the new `Reppo` algorithm in task config,
- adding any required `reppo`-specific runner config fields,
- and only touching custom runner wrappers if they need to recognize the new policy object.

The goal is to avoid changing the main training script unless a hard compatibility gap appears.

## Implementation Order

### Phase 1: Model and utility layer
1. Add a squashed-Gaussian policy implementation in `rsl_rl`.
2. Add a distributional critic implementation in `rsl_rl`.
3. Reuse `rsl_rl.modules.EmpiricalNormalization` where possible.
4. Add any helper math needed for the relative-entropy target / value binning.

### Phase 2: Algorithm layer
1. Add `rsl_rl.algorithms.reppo.Reppo`.
2. Make the algorithm build and own the actor, critic, optimizers, and normalizers.
3. Mirror the data flow from the reference implementation:
   - rollout collection,
   - bootstrapped target computation,
   - critic update,
   - actor update,
   - checkpoint save/load.

### Phase 3: Runner compatibility
1. Verify `OnPolicyRunner` can drive `Reppo` without changes.
2. Only add a new runner if the algorithm cannot fully satisfy the existing runner contract.
3. Preserve checkpointing and logging behavior.

### Phase 4: `mjlab2` integration
1. Point selected `mjlab2` tasks at the `Reppo` algorithm.
2. Keep the task registry and wrapper flow intact.
3. Update only the task configs that need new hyperparameters or model class names.

### Phase 5: Validation
1. Add a focused unit test for the new policy/model primitives.
2. Add a short runner smoke test for `Reppo`.
3. Verify `mjlab2` can still construct its existing PPO tasks unchanged.
4. Verify at least one task can train with the new `Reppo` path.

## Non-Goals
- Do not vendor the reference `reppo` repository into `mjlab2`.
- Do not introduce a parallel training entrypoint in `mjlab2` unless the existing runner contract cannot support the new algorithm.
- Do not refactor unrelated PPO or distillation code while porting `Reppo`.

## Risk Areas
- The squashed action distribution must preserve correct log-probabilities and deterministic export behavior.
- The distributional critic must stay numerically stable under the value target transform.
- Custom `mjlab2` ONNX exporters may need a thin compatibility shim if they assume PPO-specific actor internals.
- Multi-GPU behavior should remain rank-safe for checkpointing and logging.

## Definition of Done
- `rsl_rl` exposes a `Reppo` implementation that can be resolved through the same `class_name` mechanism as existing algorithms.
- `mjlab2` can opt into `Reppo` with minimal config changes.
- Existing PPO and distillation behavior in `rsl_rl` is unchanged.
- The new code has at least one smoke test covering construction and a short training step.
