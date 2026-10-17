# SGER残差Adam更新方向诊断执行说明

研究主体：**SGER-CasMVSNet（自几何与边缘细化残差-CasMVSNet网络）**。日期：2026-09-20。

针对Epoch12两步对照中“主体梯度恢复、验证提案误差仍增大，且第二步残差头梯度与实际更新内积为正”的现象，新增独立入口`experiment15_momentum_diagnose.py`。目的为检查SGER残差优化方向；不改变生产网络、正式训练默认配置或原有实验入口。

## 本轮固定条件和唯一变量

从指定两步实验的`leaky_relu/step_0001.ckpt`恢复，使用原训练清单第二个batch（索引23770、16086）。三组均为LeakyReLU(0.01)、legacy_clamp，固定Epoch12调度、原学习率、全部损失及主干冻结模式。

| 组别 | 操作 | 更新次数 |
|---|---|---:|
| keep | 完整保留Adam状态 | 1 |
| keep_repeat | 与keep相同，确定性重复 | 1 |
| zero_residual_exp_avg | 仅将ResidualDepthHead参数的一阶动量exp_avg置零 | 1 |

B组保留二阶动量exp_avg_sq、Adam步数、学习率、权重和其他分支全部状态。SGER其他可训练分支仍按原完整损失更新；“仅清零残差头动量”不等于“仅训练残差头”。程序核验实际状态差异只发生在允许的一阶动量条目。

原快照未保存完整随机状态，本轮是固定新随机状态的局部配对试验，不宣称精确重放历史第二步。原step1参数、BN缓冲区、优化器状态会与原实验`step1_update.json`指纹核对；同时核对原实验完成状态、源码哈希、快照配置及有效调度。

## 同batch测量与BN控制

每组先执行一次实际训练前向、反向、Adam更新，再对同一个已加载batch执行无梯度前向。第二次前向临时恢复第一次前向入口的BN缓冲区、模块模式和Python/NumPy/PyTorch随机状态，使前后差异反映参数更新。测量结束恢复一次真实训练前向产生的状态，不积累第二次测量的BN更新。

检查两组更新前前向和梯度精确一致；keep重复组更新及更新后前向也必须精确一致。所有阶段保护raw与置信度、主干参数/BN/优化器状态。非有限值、空有效掩码、残差越界、优化器覆盖遗漏、配对不一致会失败并保留诊断输出。

当前只测量该训练batch，没有重新评估固定验证集。因此同batch改善不能直接表述为验证改善。后续需根据本次证据决定短程验证对照。

## 执行命令

在项目根目录，使用原PyTorch/CUDA环境：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 PYTHONPATH=. python -u experiment15_momentum_diagnose.py \
  --probe_dir outputs/experiment15_residual_recovery/probe_2steps_epoch12_20260920_214234_152286 \
  --strict_determinism
```

可选只读预检（不需要GPU，不执行训练）：

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. python -u experiment15_momentum_diagnose.py \
  --probe_dir outputs/experiment15_residual_recovery/probe_2steps_epoch12_20260920_214234_152286 \
  --preflight_only
```

默认新输出目录为`outputs/experiment15_residual_recovery/momentum_epoch12_时间戳/`。`--output_dir`可指定新目录，已有目录会拒绝使用。原始快照和报告只读，每组另存`after.ckpt`，不自动进入200步或正式训练。

## 结果判读

先检查`summary.json`的`status=completed`、`checks_passed=true`和`input_files_unchanged=true`。这些是工程检查，不是质量通过结论。

每组`metrics.json`及summary中的`runs`包含：

- `before`、`after`：同batch完整损失、残差目标损失、有界目标绝对误差、提案收益、方向准确率、raw/refined指标、门控计数和激活统计。
- `after_minus_before`：误差和损失负值表示改善；proposal_gain_mean正值表示提案收益改善。计数差值仅作记录，不作为优化目标。
- `directions.parameters`：每个可训练参数的梯度有无、零梯度、L2范数、更新范数、梯度与更新内积、余弦、历史一阶动量范数和新梯度动量贡献范数。
- `directions.groups.residual_head`及`all_trainable`：分别汇总残差头和全部可训练参数。正内积仅是一阶上升方向证据，应结合实际前后损失；残差头内积不能代替完整参数内积。
- `frozen_backbone_check`、`measurement_state_restored`、B组`intervention`：保护边界和测量状态隔离证据。

若B组改善实际残差目标损失而A组恶化，支持进一步固定其余条件做动量策略短程对照，但不能直接认定长期策略有效。若两组同batch均改善而验证仍差，继续检查纠错方向分布、正向提案偏置及训练/验证模式影响。若两组均恶化，分析更新幅度和完整损失各项梯度冲突。不要同步更换BCE、扩大残差范围或降低margin。

## 本地验证

```bash
OMP_NUM_THREADS=1 PYTHONPATH=.:tests python -m unittest discover -s tests -p 'test_momentum_diagnose.py'
```

使用真实生产CascadeMVSNet及损失的32×32 CPU合成输入，覆盖三组单步、仅一阶动量改动、更新前梯度一致、重复轨迹、主干保护、BN/RNG测量入口一致、BN只保留一次更新、源快照只读及快照篡改拒绝。已对用户指定的真实step1快照执行只读预检；真实DTU/GPU诊断需运行上述命令。本轮不提供真实精度改善结论。
