# SGER-CasMVSNet Architecture Diagram Design

## Objective

Create a publication-ready, fully editable SVG architecture diagram for the proposed **SGER-CasMVSNet (Self-Geometric Edge-aware Refinement CasMVSNet)**. The figure must accurately represent the active three-stage CasMVSNet backbone and clearly mark the independent SGER blocks as a proposed extension rather than an already implemented inference component.

## Scope and Accuracy Boundary

The active repository implements the standard cascade depth pipeline, one `NormalHead` per stage, and training-time self-geometric and edge-aware losses. It does not currently enable an independent learnable refinement block (`refine=False`). The diagram therefore distinguishes:

- implemented backbone behavior: FPN features, depth-range sampling, homography warping, variance cost volume, 3D cost regularization, probability/depth regression, confidence estimation, and per-stage normal prediction;
- proposed behavior: one hybrid SGER block after every cascade stage, whose refined depth drives the next stage's local depth sampling.

No source-code changes to the network are included in this task.

## Figure Structure

The SVG uses a landscape, approximately 16:9 canvas and a left-to-right reading direction.

### Main Cascade Flow

The upper portion contains:

1. multi-view images and stage-keyed camera projection matrices;
2. a shared FPN feature extractor producing 1/4, 1/2, and full-resolution features;
3. three cascade stages with default depth hypothesis counts 48, 32, and 8;
4. one compact proposed SGER block after each stage;
5. final refined depth, surface normal, and confidence outputs.

Each stage summarizes the same internal sequence:

`Depth Sampling -> Homography Warping -> Variance Cost Volume -> 3D Cost Regularization -> Softmax and Depth Regression`

The stage also exposes photometric confidence. Its reference feature, depth, and confidence feed the stage-specific `NormalHead`.

The refined depth from SGER at stage 1 or stage 2 is resized and used as the center of the next stage's narrower depth range. Stage 3 SGER produces the final refined depth.

### Shared SGER Detail Inset

The lower central inset expands the shared SGER design once rather than repeating its internals three times. Its inputs are stage depth `D_s`, predicted normal `N_s`, photometric confidence `C_s`, and the resized reference image `I_ref,s`.

The self-geometric branch derives depth gradients, curvature, and depth-derived normals. The edge-aware branch derives image edges. These cues and confidence define:

- Region A: structural or discontinuity regions represented by a hard, high-emphasis gate;
- Region B: continuous surface regions represented by a confidence- and edge-aware soft gate.

The fused gate `G_s` modulates a CNN-predicted residual `Delta D_s`. The proposed refined output is:

`D_tilde_s = D_s + G_s * Delta D_s`

The figure labels the inset and per-stage blocks **Proposed SGER** and uses dashed correspondence lines between the compact blocks and the shared inset.

### Training Objectives

A blue-gray footer groups the multi-stage objectives:

- smooth L1 depth supervision;
- depth-normal consistency;
- normal smoothness;
- dual-region curvature continuity;
- edge-aware depth smoothness.

Dashed arrows indicate supervision rather than inference-time tensor flow.

## Visual System

The figure uses a white background and an exclusively blue color family. Distinct lightness and saturation levels separate semantic groups while remaining legible in grayscale:

- pale blue: inputs and camera data;
- blue-green tint: FPN features;
- medium blue: cascade depth stages;
- indigo-blue: normal and self-geometric paths;
- saturated deep blue: Region A;
- low-saturation cyan-blue: Region B;
- navy: refined outputs and primary labels;
- blue-gray: losses and annotations.

Solid arrows represent forward data flow. Dashed arrows represent supervision, shared-module correspondence, or conceptual/proposed connections. Stage feature planes increase in displayed size from stage 1 to stage 3 to reinforce the 1/4, 1/2, and full-resolution progression.

## Editability Requirements

The deliverable is a standalone SVG composed only of native vector elements:

- editable `<text>` labels;
- grouped and named modules using stable `id` attributes;
- vector paths, lines, polygons, and rounded rectangles;
- reusable arrow markers and style definitions;
- no embedded raster images;
- no text converted to outlines;
- compatibility with Inkscape, Adobe Illustrator, and standards-compliant browsers.

## Validation

Before delivery:

1. parse the SVG as XML;
2. confirm that all expected named groups exist;
3. confirm there are no `<image>` elements or embedded raster data;
4. render a PNG preview when a local SVG renderer is available and inspect it for clipping, overlaps, arrow routing, and text legibility;
5. report the output path and validation results.

