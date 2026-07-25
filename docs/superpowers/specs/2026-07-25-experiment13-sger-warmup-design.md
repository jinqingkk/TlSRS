# 实验13 SGER渐进启用设计

## 目标

在保持从零训练和Stage3-only SGER-Lite结构不变的前提下，渐进启用Stage3 residual及其相关损失，避免实验12中raw backbone、NormalHead、gate和ResidualHead从训练开始即强耦合，验证是否能消除Epoch 0–6深度误差接近80的早期失稳。

## 实验边界

实验13只增加SGER warm-up，不同时加入gate目标预算、residual target监督或新的confidence模型。

保持以下实验12配置：

```text
use_sger_lite=True
sger_max_residual_ratio=0.25
raw_depth_loss_weight=1.0
refined_depth_loss_weight=1.0
residual_loss_weight=0.01
gate_loss_weight=0.001
safe_refine_loss_weight=0.1
safe_refine_margin=0
freeze_backbone_epochs=0
backbone_lr_scale=1.0
```

不加载预训练checkpoint。Stage1和Stage2不运行NormalHead或SGERBlock，也不接收refined depth反馈。

## Warm-up调度

新增训练参数：

```text
sger_warmup_start_epoch=3
sger_warmup_end_epoch=6
```

两个epoch参数均为非负整数，且`end_epoch >= start_epoch`。调度定义为：

```text
epoch < start: beta = 0
epoch >= end: beta = 1
其他情况:
beta = (epoch - start + 1) / (end - start + 1)
```

默认得到：

```text
Epoch 0–2: beta=0
Epoch 3: beta=0.25
Epoch 4: beta=0.50
Epoch 5: beta=0.75
Epoch >=6: beta=1.00
```

当`start_epoch == end_epoch`时，调度退化为该epoch硬开启：之前为0，从该epoch起为1。

## 模型前向

`CascadeMVSNet`新增非参数运行状态`self.sger_residual_scale`，默认值为1.0，因此独立推理与旧checkpoint加载继续完整启用SGER。新增方法：

```text
set_sger_residual_scale(scale)
```

只接受`[0,1]`内的有限浮点数。

SGERBlock保持原输出和参数结构不变。Cascade接收其输出后计算：

```text
effective_residual = beta * depth_residual
depth_refined = depth_raw + effective_residual
effective_residual_ratio = beta * residual_ratio
```

对外的：

```text
depth_residual
residual_ratio
depth_refined
depth
```

全部表示当前beta下实际生效的结果。`geometry_gate`保持原始门控值，不乘beta，以便即使beta为0也能观察gate自身是否饱和。Stage输出和顶层输出新增标量tensor：

```text
sger_residual_scale
```

不新增模型参数，不改变state dict键。

## 损失调度

训练循环在每个epoch开始时计算beta，同时：

1. 调用模型的`set_sger_residual_scale(beta)`；
2. 将beta传给`cas_mvsnet_loss`。

损失内部仅对含有`depth_raw`的SGER stage使用以下有效权重：

```text
effective_refined_weight = beta * refined_depth_loss_weight
effective_residual_weight = beta * residual_loss_weight
effective_gate_weight = beta * gate_loss_weight
effective_safe_weight = beta * safe_refine_loss_weight
```

raw depth loss始终使用原`raw_depth_loss_weight`，不乘beta。Normal、curvature、edge-aware smooth和depth-normal consistency保持原逻辑，不乘beta。

当beta为0时：

- `depth_refined == depth_raw`；
- refined depth loss仍可计算和记录，但有效权重为0；
- residual、gate、safe-refine loss仍可计算和记录，但对total loss贡献为0；
- raw主干和原有几何监督正常训练。

验证阶段使用当前训练epoch的beta，以观察当时实际生效的网络。`test.py`不设置scale，因此默认beta为1。

## 日志

新增或保证输出：

```text
sger_residual_scale
effective_refined_depth_loss_weight
effective_residual_loss_weight
effective_gate_loss_weight
effective_safe_refine_loss_weight
stage3/raw_depth_loss
stage3/refined_depth_loss
stage3/raw_to_refined_error_delta
stage3/refined_improved_pixel_ratio
stage3/mean_abs_depth_residual
stage3/geometry_gate_mean
```

`refined_improved_pixel_ratio`定义为有效GT像素中：

```text
abs(depth_refined - GT) < abs(depth_raw - GT)
```

的比例。相等像素不计为改善，因此beta为0时该比例为0。

所有有效权重和beta以当前loss设备上的标量tensor写入extra metrics，兼容现有TensorBoard及分布式scalar归约。

## Checkpoint与恢复训练

beta不是模型参数，不进入state dict。checkpoint继续保存epoch、model和optimizer，模型参数键完全兼容实验12。

恢复训练时：

```text
start_epoch = checkpoint_epoch + 1
```

训练循环根据新epoch重新计算beta，不依赖checkpoint保存运行时scale。

为了避免训练与推理语义混淆，checkpoint保存内容新增非模型元数据：

```text
sger_warmup_start_epoch
sger_warmup_end_epoch
```

恢复训练时若checkpoint包含这两个字段且与当前CLI不一致，应报错；旧checkpoint没有字段时允许按当前CLI运行。

## 测试

采用TDD覆盖：

- 默认调度0、0.25、0.5、0.75、1；
- start等于end的硬开启行为；
- 非法epoch范围；
- 模型scale为0、0.5、1时的depth和residual关系；
- 非法scale；
- scale不进入state dict；
- beta为0时只有raw depth和原几何损失贡献total loss；
- beta为0.5与1时四类SGER损失按比例缩放；
- improved pixel ratio；
-训练循环同时设置模型scale和loss scale；
- CLI默认值、checkpoint元数据及恢复配置校验；
- Stage3-only结构和原有推理默认行为不回归。

## 成功标准

代码层面：

- warm-up公式、前向scale和损失scale一致；
- Stage1/Stage2保持raw；
- 旧checkpoint参数兼容；
- 独立推理默认beta为1；
- focused及全量测试通过。

实验层面：

- Epoch 0–3不再出现abs depth error接近80；
- beta提升时depth loss无灾难性跳变；
- Epoch 6以后记录完整SGER行为；
- 最终必须比较raw+raw、refined+raw、refined+calibrated三组结果。
