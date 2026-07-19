# Experiment 12 SGER-Lite Safe Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the full Experiment 12 SGER-Lite convergence recommendations while retaining Stage3-only refinement and conservative raw-depth/raw-confidence fusion defaults.

**Architecture:** Extend the existing `cas_mvsnet_loss` with gate sparsity and raw-versus-refined safety penalties, then construct two optimizer groups for SGER-Lite and backbone parameters. Keep the model topology unchanged, soften inference confidence calibration with configurable alpha and floor, and verify every behavior through focused tests before changing production code.

**Tech Stack:** Python 3, PyTorch, NumPy, pytest, existing CasMVSNet training and inference entry points.

---

## File Structure

- Modify `tests/test_sger_refinement.py`: behavioral regression tests for losses, optimizer grouping, freeze scheduling, confidence calibration, and CLI integration.
- Modify `models/module.py`: gate sparsity and safe-refinement loss calculations and metrics.
- Modify `train.py`: Experiment 12 defaults, loss argument forwarding, SGER-Lite optimizer grouping, and learning-rate logging.
- Modify `test.py`: Experiment 12 residual bound and softened confidence calibration configuration.

No model parameter file changes are required because Experiment 12 retains the current Stage3 NormalHead and SGERBlock topology.

### Task 1: Gate Sparsity and Safe Refinement Losses

**Files:**
- Modify: `tests/test_sger_refinement.py`
- Modify: `models/module.py:851-1030`

- [ ] **Step 1: Write failing gate-loss tests**

Add tests that use a partially valid mask and verify only valid gate values are averaged:

```python
def test_sger_gate_loss_uses_only_valid_pixels():
    depth = torch.zeros(1, 2, 2, requires_grad=True)
    outputs = {"stage1": {
        "depth": depth,
        "geometry_gate": torch.tensor([[[0.2, 0.4], [0.8, 1.0]]]),
    }}
    depth_gt = {"stage1": torch.zeros(1, 2, 2)}
    mask = {"stage1": torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])}

    total, _, extra = _unpack_loss(cas_mvsnet_loss(
        outputs, depth_gt, mask, dlossw=[1.0],
        gate_loss_weight=0.5, return_extra=True))

    assert torch.allclose(extra["stage1/gate_loss"], torch.tensor(0.3))
    assert torch.allclose(total, torch.tensor(0.15))
```

- [ ] **Step 2: Write failing safe-refinement tests**

Add one test where refined depth improves and one where it regresses:

```python
def test_safe_refine_loss_penalizes_only_regressions():
    raw = torch.tensor([[[2.0, 2.0]]])
    refined = torch.tensor([[[1.0, 3.0]]], requires_grad=True)
    outputs = {"stage1": {"depth": refined, "depth_raw": raw}}
    depth_gt = {"stage1": torch.ones(1, 1, 2)}
    mask = {"stage1": torch.ones(1, 1, 2)}

    total, _, extra = _unpack_loss(cas_mvsnet_loss(
        outputs, depth_gt, mask, dlossw=[0.0],
        raw_depth_loss_weight=0.0, refined_depth_loss_weight=0.0,
        safe_refine_loss_weight=0.1, safe_refine_margin=0.0,
        return_extra=True))

    assert torch.allclose(extra["stage1/safe_refine_loss"], torch.tensor(0.5))
    assert torch.allclose(total, torch.tensor(0.05))
```

Add a separate assertion with `refined=[1.0, 2.0]` showing the safe loss is zero, and a margin assertion showing a positive configured margin is included.

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
pytest -q tests/test_sger_refinement.py -k "gate_loss or safe_refine"
```

Expected: failures because `gate_loss_weight`, `safe_refine_loss_weight`, and their metric keys are not implemented.

- [ ] **Step 4: Implement minimal loss behavior**

In `cas_mvsnet_loss`, read:

```python
gate_loss_weight = kwargs.get("gate_loss_weight", 0.0)
safe_refine_loss_weight = kwargs.get("safe_refine_loss_weight", 0.0)
safe_refine_margin = kwargs.get("safe_refine_margin", 0.0)
```

Inside the stage loop, after raw/refined depth metrics:

```python
if (safe_refine_loss_weight > 0 and depth_raw is not None):
    raw_error = (depth_raw - depth_gt).abs()
    refined_error = (depth_est - depth_gt).abs()
    safe_refine_loss = masked_mean(
        F.relu(refined_error - raw_error + safe_refine_margin), mask)
    total_loss += safe_refine_loss_weight * safe_refine_loss
    extra["{}/safe_refine_loss".format(stage_key)] = (
        safe_refine_loss.detach())

if gate_loss_weight > 0 and "geometry_gate" in stage_inputs:
    gate_loss = masked_mean(stage_inputs["geometry_gate"], mask)
    total_loss += gate_loss_weight * gate_loss
    extra["{}/gate_loss".format(stage_key)] = gate_loss.detach()
```

Do not apply stage decay to either term.

- [ ] **Step 5: Run focused and existing loss tests**

Run:

```bash
pytest -q tests/test_sger_refinement.py -k "loss"
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit the loss change**

```bash
git add models/module.py tests/test_sger_refinement.py
git commit -m "Add safe SGER refinement losses"
```

### Task 2: Experiment 12 Training Defaults and Loss Forwarding

**Files:**
- Modify: `tests/test_sger_refinement.py`
- Modify: `train.py:75-110`

- [ ] **Step 1: Write failing CLI integration assertions**

Update the entry-point source test to require:

```python
for text in (
        "--sger_max_residual_ratio', type=float, default=0.25",
        "--residual_loss_weight', type=float, default=0.01",
        "--freeze_backbone_epochs', type=int, default=8",
        "--gate_loss_weight', type=float, default=0.001",
        "--safe_refine_loss_weight', type=float, default=0.1",
        "--safe_refine_margin', type=float, default=0.0",
        '"gate_loss_weight": args.gate_loss_weight',
        '"safe_refine_loss_weight": args.safe_refine_loss_weight',
        '"safe_refine_margin": args.safe_refine_margin',
):
    assert text in train_source
```

- [ ] **Step 2: Run the integration test and verify RED**

Run:

```bash
pytest -q tests/test_sger_refinement.py::test_train_and_test_entrypoints_expose_sger_configuration
```

Expected: failure on the first missing Experiment 12 default or argument.

- [ ] **Step 3: Add CLI arguments and forward them to the loss**

Change the existing defaults and add:

```python
parser.add_argument('--sger_max_residual_ratio', type=float, default=0.25)
parser.add_argument('--residual_loss_weight', type=float, default=0.01)
parser.add_argument('--freeze_backbone_epochs', type=int, default=8)
parser.add_argument('--gate_loss_weight', type=float, default=0.001)
parser.add_argument('--safe_refine_loss_weight', type=float, default=0.1)
parser.add_argument('--safe_refine_margin', type=float, default=0.0)
```

Forward the new values from `loss_kwargs()`:

```python
"gate_loss_weight": args.gate_loss_weight,
"safe_refine_loss_weight": args.safe_refine_loss_weight,
"safe_refine_margin": args.safe_refine_margin,
```

- [ ] **Step 4: Run the integration and loss tests**

Run:

```bash
pytest -q tests/test_sger_refinement.py -k "entrypoints or loss"
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the training configuration**

```bash
git add train.py tests/test_sger_refinement.py
git commit -m "Configure experiment 12 SGER-Lite losses"
```

### Task 3: SGER-Lite Optimizer Groups

**Files:**
- Modify: `tests/test_sger_refinement.py`
- Modify: `train.py:148-160`
- Modify: `train.py:452-483`

- [ ] **Step 1: Write failing optimizer-group tests**

Load `train.py` without executing its CLI main block, or extract a directly importable helper, then test:

```python
def test_sger_lite_optimizer_groups_cover_parameters_once():
    model = CascadeMVSNet(
        ndepths=[8, 8, 8], depth_interals_ratio=[4, 2, 1],
        use_sger_lite=True)
    groups = build_optimizer_param_groups(
        model, base_lr=0.001, backbone_lr_scale=0.1)

    assert [group["lr"] for group in groups] == [0.001, 0.0001]
    grouped = [id(param) for group in groups for param in group["params"]]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(param) for param in model.parameters()}
```

Add assertions that Stage3 `normal_head` and `sger_blocks` parameters are in the first group and a `feature` parameter is in the backbone group. Add a non-Lite test asserting one group at the base learning rate.

- [ ] **Step 2: Run optimizer tests and verify RED**

Run:

```bash
pytest -q tests/test_sger_refinement.py -k "optimizer_groups"
```

Expected: failure because `build_optimizer_param_groups` does not exist.

- [ ] **Step 3: Implement parameter classification**

Add a helper next to `set_sger_lite_freeze`:

```python
def is_sger_lite_parameter(module, name):
    prefixes = (
        "normal_head.{}.".format(module.num_stage - 1),
        "sger_blocks.",
        "shared_sger.",
        "sger_feature_adapters.{}.".format(module.num_stage - 1),
    )
    return name.startswith(prefixes)


def build_optimizer_param_groups(model, base_lr, backbone_lr_scale):
    module = model.module if hasattr(model, "module") else model
    if not getattr(module, "use_sger_lite", False):
        return [{"params": list(module.parameters()), "lr": base_lr,
                 "name": "model"}]
    if not 0.0 < backbone_lr_scale <= 1.0:
        raise ValueError("backbone_lr_scale must be in (0, 1]")
    sger_params, backbone_params = [], []
    for name, param in module.named_parameters():
        target = sger_params if is_sger_lite_parameter(module, name) else backbone_params
        target.append(param)
    return [
        {"params": sger_params, "lr": base_lr, "name": "sger_lite"},
        {"params": backbone_params, "lr": base_lr * backbone_lr_scale,
         "name": "backbone"},
    ]
```

Reuse `is_sger_lite_parameter()` inside `set_sger_lite_freeze()` so grouping and freezing cannot diverge.

- [ ] **Step 4: Add the CLI scale and construct Adam from groups**

Add:

```python
parser.add_argument('--backbone_lr_scale', type=float, default=0.1)
```

Replace the current filtered iterator with:

```python
optimizer_groups = build_optimizer_param_groups(
    model, args.lr, args.backbone_lr_scale)
optimizer = optim.Adam(
    optimizer_groups, lr=args.lr, betas=(0.9, 0.999),
    weight_decay=args.wd)
```

All parameters must enter the optimizer before the freeze schedule changes
`requires_grad`.

- [ ] **Step 5: Log both current learning rates**

Build a compact value for the existing training print:

```python
lr_text = ",".join(
    "{}={:.6f}".format(group.get("name", index), group["lr"])
    for index, group in enumerate(optimizer.param_groups))
```

Change the print label from a single numeric `lr` to `lr {}` and insert
`lr_text`. This reports scheduler-adjusted values while preserving group ratios.

- [ ] **Step 6: Run optimizer, freeze, and source integration tests**

Run:

```bash
pytest -q tests/test_sger_refinement.py -k "optimizer_groups or freeze or entrypoints"
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit optimizer grouping**

```bash
git add train.py tests/test_sger_refinement.py
git commit -m "Add SGER-Lite backbone learning-rate group"
```

### Task 4: Soft Residual Confidence Calibration

**Files:**
- Modify: `tests/test_sger_refinement.py`
- Modify: `test.py:72-125`
- Modify: `test.py:296-303`

- [ ] **Step 1: Write failing pure-function tests**

Because importing `test.py` parses required CLI arguments, parse `test.py` with
`ast.parse`, select the `FunctionDef` named
`residual_calibrated_confidence`, compile that one-node module, and execute it
with `{"np": numpy}` as the globals dictionary. Test:

```python
confidence = np.array([0.2, 0.8, 1.2], dtype=np.float32)
residual = np.array([0.0, 1.0, 100.0], dtype=np.float32)
actual = residual_calibrated_confidence(
    confidence, residual, alpha=0.25, confidence_floor=0.5)
expected_scale = 0.5 + 0.5 * np.exp(-0.25 * residual)
expected = np.clip(confidence * expected_scale, 0.0, 1.0)
np.testing.assert_allclose(actual, expected.astype(np.float32))
assert actual[0] == confidence[0]
assert actual[2] >= 0.5 * min(confidence[2], 1.0)
```

Add tests that negative alpha and floors outside `[0, 1]` raise `ValueError`.

- [ ] **Step 2: Run calibration tests and verify RED**

Run:

```bash
pytest -q tests/test_sger_refinement.py -k "calibrated_confidence"
```

Expected: failure because the function lacks alpha/floor arguments and still uses the aggressive formula.

- [ ] **Step 3: Add inference CLI controls**

Add:

```python
parser.add_argument('--confidence_residual_alpha', type=float, default=0.25)
parser.add_argument('--confidence_residual_floor', type=float, default=0.5)
```

After parsing, reject invalid values:

```python
if args.confidence_residual_alpha < 0:
    raise ValueError("--confidence_residual_alpha must be non-negative")
if not 0.0 <= args.confidence_residual_floor <= 1.0:
    raise ValueError("--confidence_residual_floor must be in [0, 1]")
```

- [ ] **Step 4: Implement and call the softened formula**

Replace the function with:

```python
def residual_calibrated_confidence(
        confidence, residual_ratio, alpha=0.25, confidence_floor=0.5):
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if not 0.0 <= confidence_floor <= 1.0:
        raise ValueError("confidence_floor must be in [0, 1]")
    residual = np.maximum(residual_ratio, 0.0)
    scale = confidence_floor + (
        1.0 - confidence_floor) * np.exp(-alpha * residual)
    calibrated = confidence * scale
    return np.clip(calibrated, 0.0, 1.0).astype(np.float32)
```

Pass `args.confidence_residual_alpha` and
`args.confidence_residual_floor` at the PFM export call.

- [ ] **Step 5: Run calibration and integration tests**

Run:

```bash
pytest -q tests/test_sger_refinement.py -k "calibrated_confidence or entrypoints"
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit confidence calibration**

```bash
git add test.py tests/test_sger_refinement.py
git commit -m "Soften SGER residual confidence calibration"
```

### Task 5: Full Verification and Documentation Consistency

**Files:**
- Verify: `models/module.py`
- Verify: `train.py`
- Verify: `test.py`
- Verify: `tests/test_sger_refinement.py`

- [ ] **Step 1: Run the focused SGER suite**

Run:

```bash
pytest -q tests/test_sger_refinement.py
```

Expected: all tests pass with zero failures.

- [ ] **Step 2: Run the full repository suite**

Run:

```bash
pytest -q
```

Expected: all repository tests pass with zero failures.

- [ ] **Step 3: Compile touched Python files**

Run:

```bash
python -m py_compile models/module.py train.py test.py tests/test_sger_refinement.py
```

Expected: exit code 0 and no output.

- [ ] **Step 4: Smoke-test training CLI help**

Run:

```bash
python train.py --help
```

Expected: exit code 0 and help text containing
`--gate_loss_weight`, `--safe_refine_loss_weight`, and
`--backbone_lr_scale`.

- [ ] **Step 5: Smoke-test inference CLI help**

Run:

```bash
python test.py --help
```

Expected: exit code 0 and help text containing
`--confidence_residual_alpha` and `--confidence_residual_floor`.

- [ ] **Step 6: Review the final diff**

Run:

```bash
git diff --check
git diff -- models/module.py train.py test.py tests/test_sger_refinement.py
```

Expected: no whitespace errors; changes are limited to the approved Experiment 12 behavior.

- [ ] **Step 7: Commit final verification fixes if needed**

```bash
git add models/module.py train.py test.py tests/test_sger_refinement.py
git commit -m "Verify experiment 12 SGER-Lite configuration"
```
