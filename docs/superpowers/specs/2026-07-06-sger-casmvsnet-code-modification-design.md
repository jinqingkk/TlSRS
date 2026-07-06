# SGER-CasMVSNet Code Modification Design

## Goal

Produce a code-level modification guide for converting the current modified CasMVSNet into a three-stage SGER-CasMVSNet. This task delivers a plan only and does not modify runtime Python code.

## Current State

The current network already implements:

- three-stage FPN/CasMVSNet depth estimation;
- per-stage depth and photometric confidence;
- one `NormalHead` per stage;
- depth-derived normals, image/depth edges, curvature, geometry weights, and Region A/B helpers;
- training-time depth-normal, normal-smoothness, curvature, and edge-aware losses.

It does not implement a learnable inference-time SGER block, refined-depth stage outputs, or refined-depth feedback into subsequent hypothesis sampling.

## File Organization

Create `models/sger_refinement.py` containing four focused modules:

- `GeometryCueExtractor`;
- `DualRegionGate`;
- `ResidualDepthHead`;
- `SGERBlock`.

Reuse geometry helpers from `models/module.py` instead of duplicating them. Avoid adapting the inactive legacy `RefineNet`, whose interface and behavior do not match SGER.

## Network Integration

Extend `CascadeMVSNet` with SGER configuration flags and one block per stage by default. After each stage's `NormalHead`, call SGER with raw depth, predicted normal, confidence, reference image, stage intrinsics, stage reference feature, and stage depth interval.

Store:

- `depth_raw`;
- `depth_refined`;
- `depth_residual`;
- `geometry_gate`;
- `region_a`;
- `region_b_weight`.

When SGER is enabled, `outputs_stage["depth"]` aliases refined depth and refined Stage1/Stage2 depth drives the next stage's local range. When disabled, outputs and numerical behavior remain compatible with the current network.

## SGER Formula

The residual head predicts a bounded residual:

`Delta D_s = alpha_s * interval_s * tanh(residual_logit_s)`.

The dual-region gate predicts `G_s` in `[0,1]`. The refined depth is:

`D_tilde_s = D_s + G_s * Delta D_s`.

Thresholded regions use detached geometry/confidence. Gradients remain enabled through predicted normal features, learnable gate heads, the residual head, and refined depth.

## Loss Integration

Extend `cas_mvsnet_loss` with:

- auxiliary raw-depth Smooth L1;
- primary refined-depth Smooth L1;
- existing depth-normal, normal-smoothness, dual-region curvature, and edge-aware losses applied to refined depth;
- normalized gated-residual magnitude regularization.

Preserve current behavior when SGER outputs are absent.

## Entry Points and Compatibility

Add CLI flags to `train.py` and `test.py` for SGER enablement, sharing, residual cap, refined feedback detach, and new loss weights. Loading pre-SGER checkpoints must use explicit missing-key reporting and deterministic zero-residual initialization. `use_sger=False` must remain the compatibility mode.

## Testing

Create `tests/test_sger_refinement.py` covering:

- tensor shapes and finite outputs;
- gate range and Region A/B exclusivity;
- bounded residual magnitude;
- zero-residual initialization;
- gradients through refined depth and learnable SGER parameters;
- detached region construction;
- SGER-disabled backward compatibility;
- refined-depth feedback selection in the cascade;
- old-checkpoint loading behavior.

## Deliverable

Create `SGER_CasMVSNet_code_modification_guide.md` in the repository root. The guide must contain exact file changes, class signatures, forward pseudocode, loss pseudocode, CLI changes, compatibility rules, phased implementation order, and verification commands. It must not claim that these changes are already implemented.

