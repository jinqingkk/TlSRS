# Improved CasMVSNet Architecture Document Design

## Goal

Create a Chinese Markdown document titled “改进的 CasMVSNet 网络架构详细说明”. It is intended for a thesis or academic method chapter and must explain the current active network from runtime inputs through coarse-to-fine depth estimation, per-stage normal prediction, geometry-aware training losses, and final outputs.

## Source of Truth

The document must be derived from the active implementation:

- `models/cas_mvsnet.py` for `CascadeMVSNet`, `DepthNet`, cascade flow, normal-head integration, and output organization;
- `models/module.py` for FPN features, homography warping, cost regularization, normal prediction, geometry helpers, and losses;
- `train.py` for model construction, default stage/loss parameters, training behavior, and logging;
- `test.py` for inference construction and output usage.

The document must state that `models/module.py` is imported by the active cascade model, while `models/module1.py` is not on the current execution path. Both training and testing instantiate `CascadeMVSNet(refine=False)`, so `RefineNet` is an available but inactive optional path.

## Audience and Style

Use a thesis-method style: begin with motivation and architecture-level reasoning, then derive data flow and formulas, and finally connect each concept to implementation. Use Chinese prose, Markdown tables, and LaTeX equations. Keep tensor shapes explicit and distinguish implemented behavior from conceptual interpretation.

## Required Sections

1. Network positioning and improvement objectives.
2. Runtime inputs, camera data, and output contract.
3. `FeatureNet` FPN structure, resolutions, and channels.
4. Three-stage coarse-to-fine depth-range sampling.
5. `DepthNet`: homography warping and variance cost-volume construction.
6. `CostRegNet`: 3D encoder-decoder regularization.
7. Probability volume, depth regression, and photometric confidence.
8. Per-stage `NormalHead` input, structure, normalization, and output.
9. End-to-end Stage1 → Stage2 → Stage3 tensor flow and default parameters.
10. Geometry sources, hard/soft geometry modulation, and Region A/B construction.
11. Smooth L1 depth, depth-normal consistency, normal smoothness, curvature, and edge-aware smoothness losses.
12. Multi-stage total objective and actual active weighting rules.
13. Gradient flow, detach behavior, and training/inference differences.
14. Output dictionary organization and top-level latest-stage aliases.
15. Improvements over the original CasMVSNet, target reconstruction problems, and implementation boundaries.
16. Class/function-to-source mapping table.

## Technical Requirements

The document must include:

- input shape `imgs: (B,N,3,H,W)`, stage-keyed projection matrices, and initial depth hypotheses;
- default FPN outputs `stage1: (B,32,H/4,W/4)`, `stage2: (B,16,H/2,W/2)`, and `stage3: (B,8,H,W)` for `base_channels=8`;
- default depth counts `48,32,8` and interval ratios `4,2,1`;
- the difference between global Stage1 sampling and later local sampling around the previous depth;
- `grad_method=detach` behavior;
- differentiable reprojection, variance aggregation, 3D cost regularization, softmax depth regression, and four-bin confidence;
- one `NormalHead` per stage using `[reference feature, depth, confidence]` and unit-length output;
- geometry masks, continuous weights, dual-region construction, and all active loss formulas;
- depth loss weights `[0.5,1.0,2.0]`;
- `1/2^(s-1)` weighting for normal smoothness, edge smoothness, and depth-normal consistency;
- the current curvature multiplier `s/2`, giving `0.5,1.0,1.5`, rather than the older shared decay description;
- the fact that multi-view reprojection is used to build the cost volume and is not an independent active reprojection loss;
- the fact that the proposed independent SEGER refinement shown in a separate conceptual diagram is not implemented in the active forward path.

## Output

Create `Improved_CasMVSNet_network_architecture.md` in the repository root. The document must stand alone and not require the earlier geometry-formula document to be understood.

## Validation

Before delivery:

1. verify all required sections and implementation-boundary statements are present;
2. verify every named class and function exists in the active source files;
3. verify all display-math delimiters are balanced;
4. scan for unfinished placeholders;
5. run `git diff --check` on the final Markdown file.

