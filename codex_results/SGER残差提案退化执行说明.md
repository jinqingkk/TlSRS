# SGER残差提案退化：代码与执行说明

日期：2026-09-20。研究主体：**SGER-CasMVSNet（自几何与边缘细化残差-CasMVSNet网络）**。

对应[研究步骤](SGER残差提案退化研究步骤.md)。本次完成代码、CPU合成验证和历史配置只读预检；未运行真实DTU审计或训练。真实运行由用户执行。

## 实现与固定变量

新增入口 `experiment15_residual_recovery.py`，支持 `audit` 和 `probe`。复用已有梯度、残差、门控审计工具，以及扩展后的 `experiment15_d_frozen_probe.py`。未修改生产网络、损失函数、正式训练默认配置。原BCE探针的默认Epoch15三组流程保留。

本轮默认Epoch12，三组依次为：

| 目录 | 末端激活 | BCE |
|---|---|---|
| relu_reference | ReLU | legacy_clamp |
| relu_repeat | ReLU | legacy_clamp |
| leaky_relu | LeakyReLU(0.01) | legacy_clamp |

唯一组间变量是残差主体末端激活。前两个ReLU保持原样。激活无可训练参数，权重、优化器分组及Adam动量从同一checkpoint恢复，不重置偏置。三组主干参数均冻结、主干BN均eval，SGER及末级法线头保持train；验证时全模型eval，验证后恢复冻结训练模式。

从已核验的Epoch12 ReLU审计报告读取该epoch模型和损失参数，并按Epoch12计算实际调度，不沿用Epoch15。checkpoint必须匹配已验证审计记录的SHA256，加载后再次检查内部epoch。默认读取 `checkpoints/experiment15_b_backbone_lr1/model_000012.ckpt`。

`--epoch 9`可作为后续复核；`--checkpoint`只用于指定已审计checkpoint的搬迁副本，哈希仍须匹配。记录checkpoint epoch、调度epoch、恢复学习率与动量、实际激活和BCE。固定调度且不增加学习率调度器。

样本清单借用 `outputs/experiment15_d/short_train/sample_manifest.json`：两步取前4个训练样本，200步取全部400个，验证使用原6个样本。只借用样本选择，不借用Epoch15权重或调度。保存原清单哈希并核对数据配置、样本索引与元数据。

## 1. 先执行无更新诊断

在项目根目录、原PyTorch/CUDA环境执行：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 PYTHONPATH=. python -u experiment15_residual_recovery.py --mode audit --epoch 12 --strict_determinism
```

该命令不执行optimizer.step，不修改输入checkpoint。审计历史Batch0–4和短程训练前两个batch，检查冻结主干后的运行条件是否仍存在退化。

每组每个batch包含5个case：

- `full_schedule`：实际完整损失及epoch调度。
- `refined_only`、`residual_target_only`：单项权重1、强制尺度1的机制检查。
- `refined_only_actual`、`residual_target_only_actual`：单项损失使用实际基础权重及当前epoch尺度。

单项case排除其他stage的无条件深度监督，使用现有审计隔离辅助函数；不是直接调用完整损失后仅修改标签。每个case恢复权重、BN与PyTorch随机状态，结束后核对模型和优化器状态未变。

默认输出：`outputs/experiment15_residual_recovery/audit_epoch12_时间戳/`。

先读 `summary.json`：

- `status=completed`、`paired_checks_passed=true`表示采集和配对检查完成，**不表示退化已解决**。
- ReLU重复组的前向与逐参数梯度比较应全部exact；LeakyReLU允许SGER差异，但raw、置信度、输入及随机状态不能出现受保护差异。
- `findings`保留各case的缺失/零关键梯度等诊断结果。基准退化正是研究对象，不能因基准出现这些发现就跳过实验组；也不能忽略它们而宣布机制通过。
- 非有限张量、优化器覆盖缺失、模型/优化器状态变动或配对一致性失败会终止并标记failed。

逐组 `historical0.json` 至 `historical4.json`、`probe0.json` 和 `probe1.json` 包含全部激活分布、bias-only差异、提案质量、门控分布及逐层梯度。`*_gradients.json`保存包含None状态的逐参数梯度指纹。

重点查看 `loss_extras.residual_diagnostics.stage3.activations` 的 `body.5.input/output`，以及 `residual_parameter_gradients` 中主体、输出权重和偏置的状态。结合 `proposal_metrics` 判断提案是仅变动，还是方向和误差收益也改善。

## 2. 诊断支持后执行两步恢复对照

先确认：冻结模式下ReLU退化现象仍存在，LeakyReLU在残差目标监督下恢复有限的主体梯度，且配对和数值检查正常。然后执行：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 PYTHONPATH=. python -u experiment15_residual_recovery.py --mode probe --epoch 12 --steps 2 --strict_determinism
```

默认输出：`outputs/experiment15_residual_recovery/probe_2steps_epoch12_时间戳/`。

三组均从原Epoch12 checkpoint恢复，执行两次更新，在第0、1、2步验证和保存快照。每步记录：

- 主干参数、BN缓冲区、优化器状态相对初始化不变，主干梯度为None。
- 所有stage raw及置信度配对一致，ReLU重复组完整训练轨迹一致。
- SGER非零梯度和相对初始化的参数变化，以及残差主体各参数的None/零/非零梯度与优化器覆盖。
- 输出形状、有限性、提案与有效残差上界。

`checks_passed=true`仅表示对照运行检查通过，不要求ReLU基准所有残差主体参数都有梯度。SGER整体仍须有非零梯度及相对初始化的参数变化。主体恢复需阅读逐层梯度和提案指标，不能用正常法线头的更新代替残差主体证据。

## 3. 两步检查通过且机制证据支持后执行200步

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 PYTHONPATH=. python -u experiment15_residual_recovery.py --mode probe --epoch 12 --steps 200 --strict_determinism
```

默认输出：`outputs/experiment15_residual_recovery/probe_200steps_epoch12_时间戳/`。

这是从原Epoch12起点重新开始的独立200步实验，不加载两步实验末尾快照。三组同批次、同顺序；第0、25、50、100、200步验证。两步实验含第1步验证，200步实验不含该节点，因此不宣称两者完整随机状态消费轨迹相同。

程序不会自动从audit推进到probe，也不会自动从两步推进到200步；各阶段是否进入下一步，应结合已产生结果判定。这三条命令不要不经检查连续运行。

## 指标与判读

`summary.json.validation`按有效像素汇总raw/refined误差、方向准确率、提案收益、残差幅度、门控precision/recall、收益正标签率及改善/恶化区域门控。额外包含：

- `proposal_target_abs_error`：相对于当前配置的有界残差目标的绝对误差。
- `proposal_minus_bias_abs_mean`：提案与相同限幅/单位变换后的bias-only输出之差。
- `gate_below_1e_minus6_ratio`、`gate_above_1_minus_1e_minus6_ratio`。
- `proposal_near_bound_ratio`：提案幅度达到配置上界99%的有效像素比例。

逐batch `metrics.json`额外记录末端激活前后正值/零值比例与空间方差、tanh前输出、输出权重贡献和偏置，以及提案在有效区域的批次方差和逐样本空间方差。这些方差不与全局像素加权误差混用。

训练指标取更新前前向，梯度取同一步反向；验证节点为对应更新完成后的状态。参数变化检查参照初始化，不等同于每个参数每一步都更新。第0步即会因切换激活产生前向差异，应与后续学习效果分开分析。

若提案变化和梯度恢复但方向或误差恶化，继续检查监督目标和优化；若提案收益提高但最终细化仍受门控限制，再进入固定激活下的BCE实验。未执行DTU融合，不把局部深度指标当作最终点云贡献。

## 输出保护与本地验证

所有运行默认创建带时间戳的新目录。`--output_dir`可指定目录，但目录必须不存在。失败或中断保留manifest、summary及已落盘证据；不自动恢复中断训练。CUDA、历史审计文件或checkpoint哈希预检失败可能发生在创建目录之前。

CPU验证命令：

```bash
OMP_NUM_THREADS=1 PYTHONPATH=.:tests python -m unittest discover -s tests -p 'test_residual_recovery.py'
OMP_NUM_THREADS=1 PYTHONPATH=. python -m unittest discover -s tests -p 'test_experiment15_d*.py'
```

覆盖真实生产CascadeMVSNet与损失的32×32合成输入，验证可控末端ReLU失活后的梯度恢复、原位激活输入捕获、实际/强制权重区分、无更新审计、重复组梯度一致、三组两步更新、raw保护、Epoch12及激活/BCE配置落盘、输入checkpoint只读，以及原BCE探针回归。真实DTU精度和GPU确定性仍须由上述用户运行验证。
