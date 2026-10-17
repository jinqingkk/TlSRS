# SGER残差提案退化 Epoch12 审计结果分析

日期：2026-09-20。研究主体：**SGER-CasMVSNet（自几何与边缘细化残差-CasMVSNet网络）**。

结论：本次审计支持进入两步恢复对照。末端 ReLU 完全失活是当前样本上残差主体零梯度的直接阻断点；仅替换为 LeakyReLU(0.01)，即可恢复有限、非零的主体梯度及具有空间变化的提案。但门控仍接近关闭，refined 尚未获得实际深度收益。不能将审计完成解释为恢复训练成功或最终精度提高。

## 证据与运行条件

- 输出：[audit_epoch12_20260920_172929_088422](../outputs/experiment15_residual_recovery/audit_epoch12_20260920_172929_088422/summary.json)。
- 命令：`CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 PYTHONPATH=. python -u experiment15_residual_recovery.py --mode audit --epoch 12 --strict_determinism`。
- checkpoint：`checkpoints/experiment15_b_backbone_lr1/model_000012.ckpt`，内部 epoch 与调度 epoch 均为 12；SHA256：`4cf0791f0305864f2a60f9b546e03ea444318df1c916e4e352a433e67a10cc17`。
- seed=1，batch_size=2，DTU训练集，3视图，512×640；PyTorch 1.13.1+cu117，RTX 4090。
- 主干参数冻结、主干 BN eval，SGER及末级法线头 train；三组使用相同恢复状态，唯一组间变量是末端激活，BCE 固定 `legacy_clamp`。未执行 optimizer.step。
- 7个batch：historical0–4（索引0–9，均来自scan2）及probe0–1（4537、9687、23770、16086，对应scan39、61、119、91）。这是有限训练样本诊断，不能代表完整验证集。
- `status=completed`、`paired_checks_passed=true`；ReLU重复组70项前向/反向比较全部exact；LeakyReLU的70项比较均有允许的SGER差异，受保护项无差异。模型及优化器状态不变检查通过。
- 3组×7batch×5case共105个case；各报告fatal_errors为空。逐层证据仍显示ReLU主体零梯度，说明空错误列表不能替代机制判断。
- 实际SGER尺度、残差目标调度尺度、收益门控调度尺度均为1；refined、残差目标、收益门控的有效基础权重分别为1、0.05、0.05。不是warm-up关闭导致残差无效。

## 残差主体：阻断点及恢复证据

ReLU在7个batch的`body.5.input`全部为负，逐batch最大值介于−0.155665与−0.023814；末端输出100%为零。有效区域的提案恒为+0.001524157，逐样本空间方差为0，与bias-only输出之差为0。

在全部35个ReLU case中，body.0/2/4的权重和偏置、output.weight共7个参数张量梯度严格为零，仅output.bias梯度非零。这些参数均requires_grad=true、梯度存在且优化器覆盖正常，因此不能归因于参数遗漏、无梯度连接或主干冻结。

LeakyReLU使负输入保留0.01斜率，末端输出零值比例变为0；提案逐样本空间方差为0.001649–0.004032。在全部35个case中，残差头8个参数张量均有有限非零梯度，但张量内部仍有部分零元素，不代表全部神经元恢复。

以下为6个body参数张量合并L2范数的跨batch范围，不含输出层：

| case | ReLU | LeakyReLU |
|---|---:|---:|
| full_schedule | 0 | 4.55e-4–3.75e-3 |
| residual_target_only_actual（权重0.05） | 0 | 4.55e-4–3.75e-3 |
| residual_target_only（强制权重1） | 0 | 9.11e-3–7.51e-2 |
| refined_only_actual / refined_only | 0 | 4.85e-16–3.37e-13 |

实际残差目标监督已经恢复可用的梯度路径；仅refined监督的梯度仍极小，与近关闭门控相符。上述直接干预证明当前末端激活的阻断作用，但不能据此还原历史训练中失活形成的原因。

## 提案质量和最终输出

下表取每个batch的full_schedule，使用stage3 GT mask>0.5，在3,506,439个有效像素上加权汇总；未重复计入其他case和ReLU重复组。深度误差沿用数据单位。

| 指标 | ReLU | LeakyReLU |
|---|---:|---:|
| raw绝对误差 | 1.940754 | 1.940754 |
| refined绝对误差 | 1.940754 | 1.940754 |
| raw−refined误差收益 | 0 | 0 |
| 提案绝对值均值 | 0.001524 | 0.142420 |
| 提案与bias-only绝对差均值 | 0 | 0.140896 |
| 提案方向准确率 | 58.8824% | 58.8824% |
| 未门控提案平均误差收益 | +0.000269 | +0.002133 |
| 有界残差目标绝对误差 | 0.380446 | 0.378582 |
| 收益门控正标签率 | 0% | 26.0760% |
| 门控均值 | 4.09e-13 | 4.09e-13 |
| 门控小于1e-6的比例 | 100% | 100% |
| 有效残差绝对值均值 | 6.23e-16 | 1.34e-14 |
| 提案达到上界99%的比例 | 0% | 0% |

两组提案在所审计有效像素上都为正，因此方向准确率完全相同。LeakyReLU恢复了幅值变化，但尚未显示正负纠错方向的改善。这里的前向变化来自替换激活，未发生学习更新。

未门控提案收益定义为raw绝对误差减去raw加提案后的绝对误差，正值表示改善：

| batch | ReLU | LeakyReLU |
|---|---:|---:|
| historical0 | +0.000039 | −0.022013 |
| historical1 | +0.000078 | −0.020509 |
| historical2 | +0.000079 | −0.021263 |
| historical3 | +0.000191 | −0.007252 |
| historical4 | +0.000392 | +0.014385 |
| probe0 | +0.000735 | +0.054106 |
| probe1 | +0.000272 | +0.007431 |

汇总收益略好，但historical0–3恶化，probe0贡献较大，不能认定提案已经稳定优于基准。应保留分batch记录，后续关注方向、过冲及样本间稳定性。

门控预测正样本数和真阳性数仍为0。ReLU目标正样本数也为0；LeakyReLU产生了收益正标签，但门控仍不响应。proposal_metrics中无定义的precision/recall应保留null语义，不能统一解读成数值0。

源码`models/module.py`的legacy BCE先将概率clamp到[1e-6,1−1e-6]。本次有效像素的门控全部低于下界，因此该BCE对这些门控概率的直接梯度被截断；完整损失下门控仍有极小梯度，不能说所有门控梯度严格为零。这解释了为什么恢复提案空间变化和收益标签，并未立即恢复最终细化，但本次没有独立BCE消融，不能宣称已验证BCE修复效果。

## 下一步

已满足执行说明的两步试验前提：冻结模式下基准退化复现，LeakyReLU在实际残差目标权重下恢复有限主体梯度，配对和数值检查通过。下一条命令为：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 PYTHONPATH=. python -u experiment15_residual_recovery.py --mode probe --epoch 12 --steps 2 --strict_determinism
```

保持本轮激活单变量和legacy BCE，检查第0/1/2步逐层梯度、实际参数变化、提案目标误差、方向及收益，同时核验raw保护和ReLU重复轨迹。应以各组第0步为起点区分激活替换的即时影响与训练所得变化。Adam历史动量可能使零当前梯度参数仍变化，参数变化不能替代主体梯度恢复证据。

两步结果支持后再判断200步实验；若提案学习改善而最终输出持续受门控限制，再固定激活单独开展BCE对照。本次仅分析已有审计并形成报告，未启动训练、修改生产代码或改写checkpoint。未执行验证集完整评估或DTU融合，Acc.、Comp.、Overall及最终重建贡献均待验证。
