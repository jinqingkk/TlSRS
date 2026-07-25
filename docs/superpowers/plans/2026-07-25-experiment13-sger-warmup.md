# 实验13 SGER渐进启用实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为Stage3-only SGER-Lite增加按epoch同步缩放residual与SGER损失的warm-up，解决从零训练早期失稳，并补齐实验13诊断指标与checkpoint配置元数据。

**Architecture:** `train.py`负责计算epoch到beta的纯函数并在每个epoch设置模型状态；`CascadeMVSNet`只负责把beta应用到实际残差；`cas_mvsnet_loss`把同一beta应用到四类SGER损失。beta不是模型参数，独立推理默认1.0，恢复训练根据epoch重建。

**Tech Stack:** Python 3.7、PyTorch、现有直接运行式测试脚本。

---

## 文件职责

- Modify `models/cas_mvsnet.py`: 保存并验证运行时residual scale，缩放Stage3实际残差。
- Modify `models/module.py`: 同步缩放SGER相关损失并输出有效权重与改善比例。
- Modify `train.py`: CLI、epoch调度、模型状态设置、loss参数转发、checkpoint元数据和恢复校验。
- Modify `tests/test_sger_refinement.py`: 覆盖调度、模型前向、损失、CLI和checkpoint集成。

### Task 1: Epoch Warm-up纯函数

**Files:**
- Modify: `tests/test_sger_refinement.py`
- Modify: `train.py`

- [ ] **Step 1: 写失败测试**

通过现有AST helper加载`compute_sger_warmup_scale`，测试：

```python
assert [compute_sger_warmup_scale(e, 3, 6) for e in range(8)] == [
    0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
assert compute_sger_warmup_scale(2, 3, 3) == 0.0
assert compute_sger_warmup_scale(3, 3, 3) == 1.0
```

并断言负start、负end和`end < start`抛出`ValueError`。

- [ ] **Step 2: 运行并确认RED**

```bash
PYTHONPATH=. python tests/test_sger_refinement.py
```

预期因helper不存在而失败。

- [ ] **Step 3: 实现纯函数与CLI**

在`train.py`增加：

```python
parser.add_argument('--sger_warmup_start_epoch', type=int, default=3)
parser.add_argument('--sger_warmup_end_epoch', type=int, default=6)

def compute_sger_warmup_scale(epoch_idx, start_epoch, end_epoch):
    if start_epoch < 0 or end_epoch < 0:
        raise ValueError("SGER warm-up epochs must be non-negative")
    if end_epoch < start_epoch:
        raise ValueError("SGER warm-up end must be >= start")
    if epoch_idx < start_epoch:
        return 0.0
    if epoch_idx >= end_epoch:
        return 1.0
    return float(epoch_idx - start_epoch + 1) / float(
        end_epoch - start_epoch + 1)
```

- [ ] **Step 4: 运行测试并确认GREEN**

```bash
PYTHONPATH=. python tests/test_sger_refinement.py
```

预期所有测试通过。

- [ ] **Step 5: 提交**

```bash
git add train.py tests/test_sger_refinement.py
git commit -m "Add SGER warmup schedule"
```

### Task 2: 模型前向Residual Scale

**Files:**
- Modify: `tests/test_sger_refinement.py`
- Modify: `models/cas_mvsnet.py`

- [ ] **Step 1: 写失败测试**

构造Lite模型，给ResidualHead非零bias，并分别设置scale 0、0.5、1，断言：

```python
scale0["depth_refined"] == scale0["depth_raw"]
scale0["depth_residual"] == 0
scale05["depth_residual"] == 0.5 * scale1["depth_residual"]
scale05["depth_refined"] == scale05["depth_raw"] + scale05["depth_residual"]
```

测试默认scale为1、非法scale抛错，并断言：

```python
"sger_residual_scale" not in model.state_dict()
```

- [ ] **Step 2: 运行并确认RED**

```bash
PYTHONPATH=. python tests/test_sger_refinement.py
```

预期因`set_sger_residual_scale`不存在而失败。

- [ ] **Step 3: 实现模型状态和缩放**

构造函数设置：

```python
self.sger_residual_scale = 1.0
```

新增：

```python
def set_sger_residual_scale(self, scale):
    scale = float(scale)
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("sger residual scale must be finite and in [0, 1]")
    self.sger_residual_scale = scale
```

在cascade接收SGER输出后使用：

```python
effective_residual = (
    self.sger_residual_scale * sger_outputs["depth_residual"])
outputs_stage["depth_residual"] = effective_residual
outputs_stage["residual_ratio"] = (
    self.sger_residual_scale * sger_outputs["residual_ratio"])
outputs_stage["depth_refined"] = depth_raw + effective_residual
outputs_stage["depth"] = outputs_stage["depth_refined"]
outputs_stage["sger_residual_scale"] = depth_raw.new_tensor(
    self.sger_residual_scale)
```

gate等非残差输出保持原值。

- [ ] **Step 4: 运行并确认GREEN**

```bash
PYTHONPATH=. python tests/test_sger_refinement.py
```

- [ ] **Step 5: 提交**

```bash
git add models/cas_mvsnet.py tests/test_sger_refinement.py
git commit -m "Scale effective SGER residuals"
```

### Task 3: SGER损失同步Warm-up

**Files:**
- Modify: `tests/test_sger_refinement.py`
- Modify: `models/module.py`
- Modify: `train.py`

- [ ] **Step 1: 写失败测试**

使用固定raw/refined、residual_ratio和gate，分别传入`sger_loss_scale=0`、`0.5`和`1`。隔离其他损失后断言：

```python
effective_refined_depth_loss_weight == beta * base_refined_weight
effective_residual_loss_weight == beta * base_residual_weight
effective_gate_loss_weight == beta * base_gate_weight
effective_safe_refine_loss_weight == beta * base_safe_weight
```

beta为0时total只含raw depth监督；beta为0.5时四类SGER贡献是beta为1的一半。测试非法beta抛错。

- [ ] **Step 2: 运行并确认RED**

```bash
PYTHONPATH=. python tests/test_sger_refinement.py
```

- [ ] **Step 3: 实现有效权重**

`cas_mvsnet_loss`读取和验证：

```python
sger_loss_scale = float(kwargs.get("sger_loss_scale", 1.0))
if not math.isfinite(sger_loss_scale) or not 0.0 <= sger_loss_scale <= 1.0:
    raise ValueError("sger_loss_scale must be finite and in [0, 1]")
effective_refined_weight = sger_loss_scale * refined_depth_loss_weight
effective_residual_weight = sger_loss_scale * residual_loss_weight
effective_gate_weight = sger_loss_scale * gate_loss_weight
effective_safe_weight = sger_loss_scale * safe_refine_loss_weight
```

只在stage包含`depth_raw`时使用有效权重；raw权重不缩放。将四个有效权重和scale写入`extra`标量tensor。

- [ ] **Step 4: 增加改善像素比例**

在raw/refined比较处加入：

```python
improved = (
    (depth_est - depth_gt).abs()
    < (depth_raw - depth_gt).abs())
extra["{}/refined_improved_pixel_ratio".format(stage_key)] = (
    masked_mean(improved.float(), mask).detach())
```

- [ ] **Step 5: 转发训练scale**

`loss_kwargs()`增加：

```python
"sger_loss_scale": getattr(args, "current_sger_warmup_scale", 1.0),
```

- [ ] **Step 6: 运行并确认GREEN**

```bash
PYTHONPATH=. python tests/test_sger_refinement.py
```

- [ ] **Step 7: 提交**

```bash
git add models/module.py train.py tests/test_sger_refinement.py
git commit -m "Warm up SGER refinement losses"
```

### Task 4: 训练循环集成

**Files:**
- Modify: `tests/test_sger_refinement.py`
- Modify: `train.py`

- [ ] **Step 1: 写失败测试**

AST加载并测试helper：

```python
set_sger_warmup_state(model, args, epoch_idx)
```

验证它：

- Lite模式设置模型scale和`args.current_sger_warmup_scale`；
- 非Lite模式固定为1；
- DataParallel wrapper正确转发；
- 默认Epoch 3得到0.25。

- [ ] **Step 2: 运行并确认RED**

```bash
PYTHONPATH=. python tests/test_sger_refinement.py
```

- [ ] **Step 3: 实现并接入epoch循环**

新增：

```python
def set_sger_warmup_state(model, args, epoch_idx):
    scale = 1.0
    if args.use_sger_lite:
        scale = compute_sger_warmup_scale(
            epoch_idx,
            args.sger_warmup_start_epoch,
            args.sger_warmup_end_epoch)
    module = model.module if hasattr(model, "module") else model
    module.set_sger_residual_scale(scale)
    args.current_sger_warmup_scale = scale
    return scale
```

每个训练epoch在train/eval前调用一次并打印beta。`test()`和`profile()`没有epoch调度时设置1。

- [ ] **Step 4: 运行并确认GREEN**

```bash
PYTHONPATH=. python tests/test_sger_refinement.py
```

- [ ] **Step 5: 提交**

```bash
git add train.py tests/test_sger_refinement.py
git commit -m "Integrate SGER warmup into training"
```

### Task 5: Checkpoint元数据与恢复校验

**Files:**
- Modify: `tests/test_sger_refinement.py`
- Modify: `train.py`

- [ ] **Step 1: 写失败测试**

测试纯helper：

```python
validate_sger_warmup_checkpoint(checkpoint, args)
```

旧checkpoint无字段时通过；字段匹配时通过；start或end不一致时抛出`RuntimeError`。

- [ ] **Step 2: 运行并确认RED**

```bash
PYTHONPATH=. python tests/test_sger_refinement.py
```

- [ ] **Step 3: 保存并校验元数据**

checkpoint增加：

```python
'sger_warmup_start_epoch': args.sger_warmup_start_epoch,
'sger_warmup_end_epoch': args.sger_warmup_end_epoch,
```

恢复optimizer前调用校验helper。`--loadckpt`作为初始化不恢复epoch/optimizer，不强制使用其中warm-up元数据。

- [ ] **Step 4: 运行并确认GREEN**

```bash
PYTHONPATH=. python tests/test_sger_refinement.py
```

- [ ] **Step 5: 提交**

```bash
git add train.py tests/test_sger_refinement.py
git commit -m "Record SGER warmup checkpoint metadata"
```

### Task 6: 全量验证

**Files:**
- Verify: `models/cas_mvsnet.py`
- Verify: `models/module.py`
- Verify: `train.py`
- Verify: `test.py`
- Verify: `tests/test_sger_refinement.py`

- [ ] **Step 1: 运行全部直接测试**

```bash
PYTHONPATH=. python tests/test_sger_refinement.py
PYTHONPATH=. python tests/test_normal_branch.py
PYTHONPATH=. python tests/test_utils_metrics.py
```

预期全部退出码0。

- [ ] **Step 2: 编译检查**

```bash
python -m py_compile models/cas_mvsnet.py models/module.py train.py test.py tests/test_sger_refinement.py
```

- [ ] **Step 3: CLI检查**

```bash
python train.py --help | rg "sger_warmup_start_epoch|sger_warmup_end_epoch"
python test.py --help
```

- [ ] **Step 4: 模型smoke检查**

实例化Lite模型，设置0/0.5/1并确认Stage1/2无SGER字段、Stage3 residual按比例变化、默认独立推理scale为1。

- [ ] **Step 5: 差异检查**

```bash
git diff --check
git status --short
```

只允许预期代码、测试和已有缓存状态。
