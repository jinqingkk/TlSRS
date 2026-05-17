# Repository Guidelines

## Project Structure & Module Organization

This repository implements CasMVSNet training, evaluation, and data conversion utilities in Python/PyTorch. Core entry points live at the root: `train.py`, `test.py`, `train.sh`, `test.sh`, `colmap2mvsnet.py`, and `gipuma.py`; `train1.py` appears to be an alternate local training script and should be treated cautiously until its role is confirmed. Network definitions are in `models/`, with `models/cas_mvsnet.py` containing the cascade model and `models/module.py` containing shared layers and geometry utilities; `models/module1.py` is an alternate module variant. Dataset loaders and I/O helpers are in `datasets/`; DTU split files are in `lists/dtu/`. MATLAB DTU evaluation scripts are under `evaluations/dtu/`. Large local data, checkpoints, and generated outputs should stay outside versioned source or under ignored paths such as `data/`, `checkpoints/`, and `outputs/`.

## CasMVSNet Network Flow

The active cascade network is `CascadeMVSNet` in `models/cas_mvsnet.py`. In this checkout it imports layer, warping, regression, and loss helpers from `models/module1.py`; `models/module.py` contains a closely related helper set plus additional normal, curvature, and edge-aware loss utilities. Check the import in `models/cas_mvsnet.py` before changing shared model behavior, because edits in `models/module.py` will not affect the current model unless the import is switched.

Runtime inputs follow the dataset output contract used by `train.py` and `test.py`: `imgs` has shape `(B, N, 3, H, W)`, `proj_matrices` is a stage-keyed dictionary such as `stage1`, `stage2`, `stage3`, and `depth_values` is the initial per-batch depth hypothesis range. `CascadeMVSNet.forward()` first records the global depth min, max, and interval from `depth_values`, then extracts multi-scale image features for every view with `FeatureNet`.

`FeatureNet` builds a coarse-to-fine feature pyramid. With the default `arch_mode="fpn"` and three stages, `stage1` is the coarsest feature map at 1/4 image resolution with `4 * base_channels` channels, `stage2` is 1/2 resolution with `2 * base_channels` channels, and `stage3` is full resolution with `base_channels` channels. The cascade loop processes these stages in order from coarse to fine.

For each stage, `CascadeMVSNet` selects that stage's view features and projection matrices, then builds depth samples with `get_depth_range_samples()`. At `stage1`, samples come from the global input `depth_values`; at later stages, the previous stage depth is optionally detached according to `--grad_method`, resized to the current image size, and used as the center of a narrower local depth range. The number of hypotheses and interval scale are controlled by `--ndepths` and `--depth_inter_r`.

`DepthNet` performs the per-stage depth estimation. It treats view 0 as the reference view, repeats the reference feature across the depth dimension, homography-warps each source feature into the reference camera for every depth hypothesis with `homo_warping()`, and aggregates all view volumes by variance. The resulting cost volume is regularized by either a per-stage `CostRegNet` or a shared one when `--share_cr` is enabled. `CostRegNet` is a 3D encoder-decoder that outputs a single-channel cost volume.

The regularized cost volume is squeezed to `(B, D, H, W)`, optionally combined with an initial probability volume, normalized with softmax over depth, and converted to a depth map by `depth_regression()`. `DepthNet` also computes `photometric_confidence` from a local depth-probability average around the regressed depth index. Each stage returns `depth` and `photometric_confidence`; `CascadeMVSNet` stores them under `outputs["stage1"]`, `outputs["stage2"]`, `outputs["stage3"]`, and also updates top-level `outputs["depth"]` and `outputs["photometric_confidence"]` with the latest stage.

The final output depth is therefore the finest stage depth unless `refine=True` is enabled. The refinement path exists as `RefineNet`, which predicts a residual from the reference RGB image and the current depth, but the training and testing entry points instantiate `CascadeMVSNet(refine=False)` by default. Training computes multi-stage smooth L1 depth supervision through `cas_mvsnet_loss`; command-line weights such as `--dlossw` control stage weighting. The current `train.py` also passes normal, curvature, and edge-aware loss weights, but those extra arguments only affect training if the active imported loss implementation consumes them.

## Build, Test, and Development Commands

Install dependencies in an isolated environment. This checkout does not currently include a `requirements.txt`, so follow the upstream README/environment notes and keep the legacy PyTorch/CUDA stack compatible with the codebase. Common packages used by the scripts include:

```bash
pip install torch torchvision tensorboardX opencv-python scipy pillow plyfile
```

Train on DTU after setting `MVS_TRAINING` inside `train.sh`:

```bash
./train.sh 8 ./checkpoints --ndepths "48,32,8" --depth_inter_r "4,2,1" --batch_size 2
```

Run inference/fusion after setting `TESTPATH` in `test.sh` and providing a checkpoint:

```bash
./test.sh ./checkpoints/casmvsnet.ckpt --outdir ./outputs --interval_scale 1.06
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
