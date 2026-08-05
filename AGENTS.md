# Repository Guidelines

## Project Structure & Module Organization

This repository implements CasMVSNet training, evaluation, and data conversion utilities in Python/PyTorch. Core entry points live at the root: `train.py`, `test.py`, `train.sh`, `test.sh`, `colmap2mvsnet.py`, and `gipuma.py`; `train1.py` appears to be an alternate local training script and should be treated cautiously until its role is confirmed. Network definitions are in `models/`, with `models/cas_mvsnet.py` containing the cascade model and `models/module.py` containing the active shared layers, geometry utilities, loss functions, and normal helpers; `models/module1.py` is an alternate module variant that is not imported by the current cascade model. Dataset loaders and I/O helpers are in `datasets/`; DTU split files are in `lists/dtu/`, including `test2.txt` for an additional small test list. MATLAB DTU evaluation scripts are under `evaluations/dtu/`. Large local data, checkpoints, and generated outputs should stay outside versioned source or under ignored paths such as `data/`, `checkpoints/`, and `outputs/`.

## SGER-CasMVSNet Project Identity

The official Chinese full name of **SGER-CasMVSNet** is **自几何与边缘细化残差-CasMVSNet网络**. Use this exact name in project documentation, experiment reports, architecture descriptions, and research summaries. Do not redefine SGER as a generic post-processing module or omit its role when naming the proposed network.

SGER is the central research contribution of this project. CasMVSNet provides the multi-view matching, cascade depth estimation, and raw-depth backbone, while SGER provides the defining self-geometry, edge-aware, and residual-refinement mechanism. Future architecture design, loss design, training strategies, ablation studies, and improvement proposals must remain centered on making SGER effective, reliable, and measurably beneficial. A raw CasMVSNet path may be retained as a protected backbone, control group, fallback, or source of supervision, but it must not replace SGER as the final research focus.

When experiments show that the current refined output is worse than the raw output, treat that result as evidence for improving SGER's residual direction, gating, geometry/edge cues, uncertainty modeling, or optimization. Do not conclude that the project should remove or bypass SGER as its final architecture. The target is for the SGER-refined output to outperform the raw CasMVSNet output under the same evaluation protocol. Every proposed change should state explicitly which SGER component it improves and should include raw-versus-refined ablations that demonstrate SGER's independent contribution.

## CasMVSNet Network Flow

The active cascade network is `CascadeMVSNet` in `models/cas_mvsnet.py`. In this checkout it imports layer, warping, regression, normal-head, and loss helpers from `models/module.py`. Check the import in `models/cas_mvsnet.py` before changing shared model behavior, because `models/module1.py` is a nearby alternate implementation but is not active unless the import is changed.

Runtime inputs follow the dataset output contract used by `train.py` and `test.py`: `imgs` has shape `(B, N, 3, H, W)`, `proj_matrices` is a stage-keyed dictionary such as `stage1`, `stage2`, `stage3`, and `depth_values` is the initial per-batch depth hypothesis range. `CascadeMVSNet.forward()` first records the global depth min, max, and interval from `depth_values`, then extracts multi-scale image features for every view with `FeatureNet`.

`FeatureNet` builds a coarse-to-fine feature pyramid. With the default `arch_mode="fpn"` and three stages, `stage1` is the coarsest feature map at 1/4 image resolution with `4 * base_channels` channels, `stage2` is 1/2 resolution with `2 * base_channels` channels, and `stage3` is full resolution with `base_channels` channels. The cascade loop processes these stages in order from coarse to fine.

For each stage, `CascadeMVSNet` selects that stage's view features and projection matrices, then builds depth samples with `get_depth_range_samples()`. At `stage1`, samples come from the global input `depth_values`; at later stages, the previous stage depth is optionally detached according to `--grad_method`, resized to the current image size, and used as the center of a narrower local depth range. The number of hypotheses and interval scale are controlled by `--ndepths` and `--depth_inter_r`.

`DepthNet` performs the per-stage depth estimation. It treats view 0 as the reference view, repeats the reference feature across the depth dimension, homography-warps each source feature into the reference camera for every depth hypothesis with `homo_warping()`, and aggregates all view volumes by variance. The resulting cost volume is regularized by either a per-stage `CostRegNet` or a shared one when `--share_cr` is enabled. `CostRegNet` is a 3D encoder-decoder that outputs a single-channel cost volume.

The regularized cost volume is squeezed to `(B, D, H, W)`, optionally combined with an initial probability volume, normalized with softmax over depth, and converted to a depth map by `depth_regression()`. `DepthNet` also computes `photometric_confidence` from a local depth-probability average around the regressed depth index. Each stage returns `depth` and `photometric_confidence`; `CascadeMVSNet` stores them under `outputs["stage1"]`, `outputs["stage2"]`, `outputs["stage3"]`, and also updates top-level `outputs["depth"]` and `outputs["photometric_confidence"]` with the latest stage. Every active cascade stage has its own normal head. The normal branch concatenates that stage's reference feature, stage depth, and stage confidence, then predicts `outputs[stage_key]["normal"]` as a unit-length `(B, 3, H, W)` surface normal map at the stage resolution. The top-level `outputs["normal"]` is still updated by the latest stage, so in the default three-stage model it is the stage3 normal map.

The final output depth is therefore the finest stage depth unless `refine=True` is enabled. The refinement path exists as `RefineNet`, which predicts a residual from the reference RGB image and the current depth, but the training and testing entry points instantiate `CascadeMVSNet(refine=False)` by default. Training computes multi-stage smooth L1 depth supervision through `cas_mvsnet_loss`; command-line weights such as `--dlossw` control depth loss weighting for each stage.

`cas_mvsnet_loss` also consumes the normal and smoothness options passed by `train.py`. `--normal_smooth_loss_weight`, `--curv_loss_weight`, and `--edge_smooth_loss_weight` are applied to every numeric cascade stage and divided by `2 ** stage_idx`, so stage1 receives the full base weight, stage2 receives half, and stage3 receives one quarter. With the current defaults, the base weights are `0.02`, `0.005`, and `0.005`. For each stage that has photometric confidence, the loss builds a geometry mask from valid depth pixels, high-confidence pixels, and non-edge pixels according to `--depth_normal_conf_threshold` and `--edge_grad_threshold`. The curvature loss uses this geometry mask so curvature continuity is only enforced in reliable non-edge regions. The depth-normal consistency term is enabled by `--depth_normal_loss_weight`, uses the current stage's predicted normal, predicted depth, photometric confidence, reference image gradients, and camera intrinsics, and is weighted with the same `1 / 2 ** stage_idx` stage decay. The edge-aware smooth loss keeps its original image-gradient weighting. Training and validation log extra scalar metrics such as `normal_smooth_loss`, `curv_loss`, `edge_smooth_loss`, `depth_normal_loss`, `geometry_mask_ratio`, `normal_depth_cos`, `smooth_mask_ratio`, and `non_edge_depth_grad_mean`; image summaries may include the latest-stage `normal_pred` and `smooth_mask`.

## Build, Test, and Development Commands

Install dependencies in an isolated environment. This checkout does not currently include a `requirements.txt`, so follow the upstream README/environment notes and keep the legacy PyTorch/CUDA stack compatible with the codebase. Common packages used by the scripts include:

```bash
pip install torch torchvision tensorboardX opencv-python scipy pillow plyfile
```

Train on DTU after setting `MVS_TRAINING` inside `train.sh`. The shell scripts are currently stored without executable permission, so run them through `bash` unless you first restore the executable bit:

```bash
bash train.sh 8 ./checkpoints --ndepths "48,32,8" --depth_inter_r "4,2,1" --batch_size 2
```

The repository also includes `cmdexplain.txt` with a recorded one-GPU training example using `./checkpoints/cas_dtu`, `--dlossw "0.5,1.0,2.0"`, `--batch_size 2`, and `--eval_freq 3`.

Run inference/fusion after setting `TESTPATH` in `test.sh` and providing a checkpoint:

```bash
bash test.sh ./checkpoints/casmvsnet.ckpt --outdir ./outputs --interval_scale 1.06
```

Convert a COLMAP dense reconstruction to CasMVSNet input:

```bash
python colmap2mvsnet.py --dense_folder COLMAP/dense --save_folder outputs/colmap/casmvsnet
```

## Coding Style & Naming Conventions

Use Python 3 syntax compatible with the legacy PyTorch stack this project targets. Avoid adding APIs that require newer PyTorch features unless the runtime requirement is documented. Follow existing style: 4-space indentation, `snake_case` for functions/variables, `CamelCase` for `nn.Module` and `Dataset` classes, and argparse flags in lowercase with underscores. Keep tensor shape assumptions explicit with assertions or short comments. Prefer existing helpers from `utils.py`, `datasets/data_io.py`, and `models/module.py` before adding new utilities.

## Testing Guidelines

There is no standalone unit test suite in this checkout. Validate changes with the smallest feasible runtime path: import touched modules, run a short `test.py` inference on a small scene when data/checkpoints are available, or run a reduced training command with low epochs/batch size. For dataset or geometry changes, verify expected input folders (`Cameras`, `Depths`, `Rectified`, `cams`, `images`, `pair.txt`) and inspect generated `.pfm`, `.ply`, or log outputs.

## Commit & Pull Request Guidelines

Git history is unavailable in this workspace, so use concise imperative commit subjects such as `Fix DTU camera parsing` or `Add COLMAP conversion check`. Pull requests should describe the dataset/checkpoint used, commands run, expected metric or output changes, and any GPU/CUDA assumptions. Include screenshots or point-cloud inspection notes when visual reconstruction quality changes.

## Security & Configuration Tips

Do not commit datasets, downloaded checkpoints, TensorBoard logs, or generated point clouds. Keep machine-specific paths in shell scripts or documented environment variables, and avoid hard-coding private storage locations in Python modules.
