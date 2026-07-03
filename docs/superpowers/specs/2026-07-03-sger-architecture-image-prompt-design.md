# SGER-CasMVSNet Architecture Image Prompt Design

## Goal

Create a general-purpose text-to-image prompt that describes how to draw an accurate SGER-CasMVSNet architecture figure in the visual style of the supplied CasMVSNet reference image.

## Visual Reference

The reference uses a wide white canvas, academic vector-diagram styling, thin outlines, simple isometric feature/cost-volume blocks, three horizontal cascade rows, colored functional arrows, compact circular operation nodes, small depth-map outputs, and a legend along the bottom.

The prompt must reuse this visual language without copying the fruit photograph or unrelated scene content.

## Layout

Use a wide landscape composition. The left side shows multi-view image planes and a three-scale FPN. The center shows three horizontal cascade stages. Each stage contains:

`Differentiable Homography Warping -> Feature Volumes -> Variance Cost Metric -> Cost Volume -> 3D CNN -> Depth Regression`.

Each stage exposes depth and confidence. A branch combines reference feature, depth, and confidence in a `Normal Head` to produce a normal map. A compact SGER block then consumes depth, normal, confidence, resized reference image, and stage intrinsics to produce refined depth.

Refined Stage1 and Stage2 depth feed the next stage's hypothesis-plane generation through yellow feedback arrows. Stage3 refined depth is the final depth output.

## Shared SGER Inset

Expand SGER internals once in a lower inset:

- inputs `D_s`, `N_s`, `C_s`, `I_ref`, `K_s`;
- self-geometric cues: depth-derived normal, normal disagreement, depth gradient, curvature;
- edge-aware cues: image/Sobel edge and confidence reliability;
- `Region A: Hard Gate` and `Region B: Soft Weight`;
- gate fusion `G_s`;
- residual CNN producing `Delta D_s`;
- editable-looking formula `D_tilde_s = D_s + G_s * Delta D_s`.

## Visual Encoding

- white background and clean publication aesthetics;
- pale blue/blue-gray isometric cost volumes and 3D CNN blocks;
- tan feature-volume blocks;
- green warping/geometric-flow arrows and circular `W`/`M` nodes;
- yellow refined-depth/hypothesis feedback arrows;
- indigo normal and SGER branches;
- thin dark outlines, consistent perspective, generous spacing;
- English labels only;
- bottom legend for all symbols and arrow meanings.

## Prompt Deliverable

Provide:

1. one detailed English master prompt suitable for general text-to-image systems;
2. one Chinese component-by-component explanation;
3. one negative prompt preventing photorealism, decorative clutter, incorrect stage count, missing SGER/normal branches, unreadable text, broken arrows, and inconsistent perspective;
4. optional short generation guidance noting that image models may need later vector/text cleanup.

## Accuracy Boundaries

The figure represents a proposed SGER-CasMVSNet. It must not imply that the independent SGER inference module is already active in the current repository. It must show exactly three cascade stages and one SGER application per stage, while expanding the shared SGER design only once.

