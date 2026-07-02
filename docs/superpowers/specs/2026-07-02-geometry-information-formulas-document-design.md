# CasMVSNet Geometry Information and Constraints Document Design

## Goal

Create one Chinese Markdown document that explains the geometric information sources and mathematical constraints used by the modified CasMVSNet implementation. The document is intended for research-method writing: formulas use conventional academic notation, while every formula is mapped back to the active repository implementation.

## Accuracy Boundary

The document must distinguish direct implementation from conceptual interpretation. Multi-view reprojection is directly used by homography warping and variance cost-volume construction. It is a source of multi-view geometric evidence, but the active `cas_mvsnet_loss` does not define an independent reprojection-error loss. This distinction must be explicit.

The active sources are `models/cas_mvsnet.py`, `models/module.py`, and the loss defaults passed by `train.py`. Nearby alternate implementations such as `models/module1.py` are out of scope.

## Document Structure

1. Purpose and notation.
2. Geometry information sources:
   - depth probability volume and regressed depth;
   - camera-aware normals recovered from predicted depth;
   - per-stage normals predicted by `NormalHead`;
   - multi-view homography reprojection and variance aggregation;
   - probability-volume photometric confidence;
   - image gradients, Sobel edges, normalized depth gradients, and curvature.
3. Geometry constraints:
   - depth-normal consistency on valid, high-confidence, non-edge pixels;
   - normal smoothness derived from local depth gradients;
   - raw, soft-weighted, and dual-region curvature continuity;
   - edge-aware depth smoothness;
   - hard geometry masks, continuous geometry weights, and Region A/B modulation.
4. Multi-stage total objective and current default hyperparameters.
5. Formula-to-code mapping table.
6. Implementation notes and limitations.

## Formula Requirements

The document must define dimensions and symbols before use. Each source or loss section must include:

- input origin;
- mathematical formula;
- computation sequence;
- physical or geometric interpretation;
- role in reconstruction;
- corresponding implementation function.

The formulas must reflect implementation details including detached normalization means, finite differences, neighborhood stencil weights, confidence sigmoid weighting, Sobel edge magnitude, dual-region masks, and stage-dependent loss weights. The curvature stage multiplier in the current code is `0.5(s+1)` for zero-based stage index `s`; the other geometry loss terms use `1/2^s`. The document must not repeat older descriptions that assign the same decay to curvature when the active code differs.

## Output

Create `CasMVSNet_geometry_information_and_constraints.md` in the repository root. Use Chinese prose, Markdown tables, and LaTeX equations. The document must be standalone and editable, with no external links required for comprehension.

## Validation

Before delivery:

1. confirm all required sections and formulas are present;
2. confirm every named implementation function exists in the active files;
3. scan for unfinished placeholders and malformed equation delimiters;
4. verify the multi-view reprojection boundary and stage-weight distinction are stated explicitly;
5. run `git diff --check` on the final document.

