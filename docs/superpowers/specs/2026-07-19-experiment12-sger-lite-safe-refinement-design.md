# Experiment 12 SGER-Lite Safe Refinement Design

## Goal

Implement the complete Experiment 12 recommendation from
`codex_results/实验11分析.md` without expanding SGER beyond Stage3. The change
must make refinement conservative, explicitly penalize regressions relative to
raw depth, soften residual-based confidence calibration, and fine-tune the
backbone at a lower learning rate after the initial freeze period.

## Scope

The active model path remains:

```text
models/cas_mvsnet.py -> models/module.py
```

`models/module1.py` is not modified. Stage1 and Stage2 remain unchanged raw
CasMVSNet stages. Stage3 remains the only stage that runs NormalHead and
SGERBlock when `--use_sger_lite` is enabled.

The implementation changes:

- Experiment 12 training defaults.
- Two Stage3 refinement losses.
- SGER-Lite optimizer parameter groups.
- Residual-calibrated confidence behavior and its CLI controls.
- Training metrics and focused automated tests.

It does not add a new model module, change the cost volume, change depth
sampling, or feed refined depth back into the cascade.

## Experiment 12 Defaults

The SGER-Lite experiment uses:

```text
sger_max_residual_ratio = 0.25
raw_depth_loss_weight = 1.0
refined_depth_loss_weight = 1.0
residual_loss_weight = 0.01
freeze_backbone_epochs = 8
gate_loss_weight = 0.001
safe_refine_loss_weight = 0.1
safe_refine_margin = 0.0
backbone_lr_scale = 0.1
confidence_residual_alpha = 0.25
confidence_residual_floor = 0.5
```

The new loss terms are inert when their corresponding output is absent, so the
baseline CasMVSNet path is unaffected. Existing CLI flags remain overridable.

## Gate Sparsity Loss

For every stage containing `geometry_gate`, compute:

```text
L_gate = mean(geometry_gate[valid_gt_mask])
```

Add `gate_loss_weight * L_gate` directly to the total loss. Do not apply the
existing `1 / 2**stage_idx` geometry-loss decay: in SGER-Lite this loss exists
only at Stage3, and the configured weight is intended to be its effective
weight.

Log:

```text
stage3/gate_loss
stage3/geometry_gate_mean
```

The existing gate metric remains unchanged. The optimization target is to move
the mean gate from the Experiment 11 value near 0.72 toward the observed target
range 0.15–0.35 without hard-clamping the gate.

## Safe Refinement Loss

For valid ground-truth pixels:

```text
err_raw = abs(depth_raw - depth_gt)
err_refined = abs(depth_refined - depth_gt)
L_safe = mean(relu(err_refined - err_raw + safe_refine_margin))
```

Add `safe_refine_loss_weight * L_safe` directly to the total loss. With the
default zero margin, there is no penalty where refined depth is at least as
accurate as raw depth, and the loss increases only by the amount refinement is
worse.

The loss is evaluated only when both `depth_raw` and refined `depth` are
available. It uses the existing stage validity mask and must return a finite
zero when no valid comparison pixels exist, following the repository's masked
loss conventions.

Log:

```text
stage3/safe_refine_loss
stage3/raw_to_refined_error_delta
```

The existing error delta convention remains: positive means refinement
improved the mean absolute error.

## Optimizer and Freeze Schedule

When `--use_sger_lite` is enabled, create two optimizer parameter groups before
training:

1. SGER-Lite parameters at the configured base learning rate:
   - final-stage NormalHead;
   - `sger_blocks`;
   - a shared SGER block or final-stage adapter if present.
2. All remaining backbone parameters at
   `base_lr * backbone_lr_scale`.

All model parameters remain registered in the optimizer from the start.
During epochs where `epoch_idx < freeze_backbone_epochs`,
`set_sger_lite_freeze()` sets the backbone parameters to
`requires_grad=False`. At the first later epoch it restores them to
`requires_grad=True`; they then update at the backbone group's lower learning
rate. No optimizer reconstruction occurs, so optimizer and scheduler state are
preserved.

Non-Lite training retains the existing single parameter group and learning
rate behavior.

The learning-rate scheduler must preserve the ratio between parameter groups.
Training logs should expose both current group learning rates when SGER-Lite is
active.

## Confidence Calibration

Replace the aggressive calibration:

```text
confidence * exp(-residual_ratio)
```

with:

```text
scale = floor + (1 - floor) * exp(-alpha * max(residual_ratio, 0))
confidence_calibrated = clip(confidence * scale, 0, 1)
```

Defaults are `alpha=0.25` and `floor=0.5`. Validate:

```text
alpha >= 0
0 <= floor <= 1
```

Expose both values as inference CLI arguments. Continue exporting:

- `confidence/` for raw photometric confidence;
- `confidence_residual_calibrated/` for softened calibration.

Keep `--fusion_confidence_source raw` as the default. Keep raw and refined
depth exports and `--fusion_depth_source raw` as the default so the recommended
Experiment 12 evaluation remains conservative.

## Checkpoint Compatibility

No new trainable model parameters are introduced by the new losses,
calibration, or optimizer grouping. Existing Experiment 8 checkpoint
initialization behavior remains:

- missing SGER-only parameters are allowed;
- unrelated missing keys and all unexpected keys are rejected.

Native Experiment 12 checkpoints use the same model parameter names as the
current SGER-Lite implementation.

## Tests

Use test-driven development and add focused tests before production changes.

Loss tests cover:

- gate loss equals the valid-pixel gate mean times its configured weight;
- invalid pixels do not influence gate loss;
- safe loss is zero when refined depth improves or matches raw depth;
- safe loss equals the regression amount when refined depth is worse;
- margin behavior and logging keys;
- paths without SGER outputs remain unchanged.

Optimizer tests cover:

- exactly two SGER-Lite parameter groups;
- every trainable parameter appears exactly once;
- SGER-Lite parameters use base learning rate;
- backbone parameters use `backbone_lr_scale`;
- freeze and unfreeze preserve intended `requires_grad` states;
- non-Lite mode retains existing optimizer behavior.

Calibration tests cover:

- zero residual preserves confidence;
- increasing residual lowers confidence monotonically;
- calibrated confidence never falls below `floor * confidence`;
- output remains in `[0, 1]`;
- invalid alpha and floor values are rejected.

CLI/source integration tests cover all Experiment 12 defaults and forwarding of
new loss, optimizer, and calibration arguments.

Verification consists of the focused SGER tests, the full repository test
suite, Python compilation/import checks, and CLI help smoke checks for
`train.py` and `test.py`.

## Success Criteria

Implementation is complete when:

- SGER-Lite retains Stage3-only behavior.
- All Experiment 12 defaults are exposed and overridable.
- Gate and safe-refinement losses affect only eligible valid pixels and are
  logged.
- The backbone is frozen for eight epochs by default, then trains at one tenth
  of the SGER-Lite learning rate.
- Residual confidence calibration uses the softened formula with configurable
  alpha and floor.
- Raw depth and raw confidence remain the default fusion sources.
- Existing checkpoint compatibility checks remain strict outside SGER-only
  missing keys.
- Focused and full verification commands pass.
