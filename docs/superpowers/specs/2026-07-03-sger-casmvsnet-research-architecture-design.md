
# SGER-CasMVSNet Research Architecture Document Design

## Goal

Create a Chinese Markdown research proposal that explains how the current modified CasMVSNet can be extended into **SGER-CasMVSNet (Self-Geometric Edge-aware Refinement CasMVSNet)**. The document must describe a coherent future network rather than claiming that the proposed refinement module already exists in the active code.

## Current Baseline

The baseline is the current active implementation:

- three-stage CasMVSNet coarse-to-fine depth estimation;
- FPN multi-scale features;
- homography warping, variance cost volumes, and per-stage 3D cost regularization;
- depth regression and photometric confidence;
- one `NormalHead` per stage;
- training-time depth-normal, normal-smoothness, curvature, edge-aware, confidence-weighted, and Region A/B geometry constraints.

The independent learnable SGER refinement block is not currently implemented and must be labeled as proposed work.

## Proposed Architecture

Insert one SGER block after every cascade stage. Stage `s` produces raw depth `D_s`, predicted normal `N_s`, and confidence `C_s`. SGER additionally receives the resized reference image `I_ref,s`, stage intrinsics `K_s`, and optionally the reference feature map.

SGER derives:

- camera-aware depth normal;
- image/Sobel edge strength;
- normalized depth-gradient strength;
- depth curvature;
- confidence-aware reliability;
- hard Region A and soft Region B modulation.

A residual CNN predicts `Delta D_s`. A geometry-edge gate `G_s` modulates the residual:

`D_tilde_s = D_s + G_s * Delta D_s`.

`D_tilde_1` and `D_tilde_2` become the local depth-range centers for the next cascade stages. `D_tilde_3` is the final refined depth. Each stage uses the same SGER topology but independent parameters; weight sharing is reserved for ablation.

## Training Design

Retain auxiliary supervision on raw stage depth to stabilize the inherited backbone. Apply primary depth supervision to refined depth and geometry losses to refined depth plus predicted normal. Add a residual-magnitude regularizer so refinement remains local.

The proposed objective must include:

- raw multi-stage depth loss;
- refined multi-stage depth loss;
- depth-normal consistency;
- normal smoothness;
- dual-region curvature;
- edge-aware smoothness;
- residual regularization.

All region masks and confidence-derived gates must have explicit gradient-flow rules. Thresholded region construction should use detached geometry; the residual CNN and refined depth remain differentiable.

## Required Document Sections

1. Research motivation and naming.
2. Relationship to the current modified CasMVSNet.
3. Overall SGER-CasMVSNet data flow.
4. Baseline FPN/cascade/depth/normal path.
5. Per-stage SGER inputs and outputs.
6. Self-geometric cue extractor.
7. Edge-aware cue extractor.
8. Region A/B dual-region modulation.
9. Residual CNN and gated residual formula.
10. Cross-stage refined-depth feedback.
11. Complete training objective and gradient flow.
12. Training schedule and initialization strategy.
13. Inference process and output contract.
14. Suggested implementation modules and code integration points.
15. Ablation experiments and evaluation metrics.
16. Expected benefits, risks, and limitations.
17. Explicit current-versus-proposed comparison table.

## Research Rigor

The proposal must avoid presenting untested benefits as established results. Expected improvements must be phrased as hypotheses. It must identify possible failure modes including edge-texture mismatch, residual over-correction, noisy early-stage normals, threshold sensitivity, and added compute/memory cost.

The ablation plan must isolate:

- Stage3-only versus all-stage SGER;
- hard-only, soft-only, and dual-region gating;
- geometry gate without residual CNN versus complete hybrid refinement;
- shared versus independent SGER parameters;
- raw-depth auxiliary supervision and residual regularization;
- each geometry loss contribution.

## Output

Create `SGER_CasMVSNet_research_architecture.md` in the repository root. Use Chinese prose, Markdown tables, LaTeX equations, and compact text flow diagrams. The document must be standalone.

## Validation

Before delivery:

1. verify all required proposed components and research sections exist;
2. verify the current/proposed boundary is explicit;
3. verify all display-math delimiters are balanced;
4. scan for unfinished placeholders;
5. run `git diff --check` on the final document.

