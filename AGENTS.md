# AGENTS.md

## Scope

`rsl_rl` is the core RL library. In this workspace it should stay REPPO-focused:
- keep the `Reppo` implementation in `rsl_rl`,
- keep legacy PPO and distillation behavior stable,
- and keep any `mjlab2`-specific compatibility logic out of this repository.

If a change is only needed because of `mjlab2` integration, implement the adapter on the `mjlab2` side.

## Current Boundary

The current public additions are:
- `rsl_rl.models.ReppoPolicy`
- `rsl_rl.models.ReppoCritic`
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
