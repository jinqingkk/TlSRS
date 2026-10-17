# SGER残差动量25步对照执行说明

日期：2026-09-21。研究主体：**SGER-CasMVSNet（自几何与边缘细化残差-CasMVSNet网络）**。

本轮针对SGER残差优化状态，验证单batch观察到的一阶动量清零收益能否持续并推广到固定验证集。入口为`experiment15_momentum_probe.py`。不修改生产网络、损失、正式训练默认参数和已有单步诊断入口。

## 实验边界

三组均从原两步实验的`leaky_relu/step_0001.ckpt`重新恢复，不加载单步动量诊断的after.ckpt：

| 组别 | 起点处理 | 后续更新 |
|---|---|---|
| keep | 保留完整Adam状态 | 正常Adam |
| keep_repeat | 同keep | 确定性重复 |
| zero_residual_exp_avg | 仅在起点清零一次残差头exp_avg | 正常累积动量，绝不每步清零 |

保留二阶动量、Adam步数、其他分支状态及恢复的学习率。固定LeakyReLU(0.01)、legacy_clamp、Epoch12调度、完整损失及主干冻结/BN eval策略；SGER和末级法线头保持可训练。不引入scheduler。

新实验的local step0是原快照已完成1次局部更新后的状态，local step25表示再执行25次更新。训练采用原400样本清单的`[2:52]`，跳过起点已用过的首个batch；第一个batch仍为23770、16086，不重新抽样。验证复用原6个样本（scan3、scan5、scan17各两张），节点为0、1、5、10、25。

清单路径及哈希由原两步实验manifest读取。预检核验源码、来源快照指纹、有效调度、原训练前缀、验证选择及样本清单配置；运行时进一步核对dataset metadata及组间输入张量。

这是新设种子的局部配对实验，不宣称重放原实验的全部随机状态。它也不是正式联合训练或DTU点云评估。

## 执行

在项目根目录和原CUDA环境运行：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 PYTHONPATH=. python -u experiment15_momentum_probe.py \
  --probe_dir outputs/experiment15_residual_recovery/probe_2steps_epoch12_20260920_214234_152286 \
  --strict_determinism
```

可选只读预检，不执行训练：

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. python -u experiment15_momentum_probe.py \
  --probe_dir outputs/experiment15_residual_recovery/probe_2steps_epoch12_20260920_214234_152286 \
  --preflight_only
```

默认输出`outputs/experiment15_residual_recovery/momentum_25steps_epoch12_时间戳/`，也可用`--output_dir`指定不存在的新目录。固定25步，不接受未经实现的--steps配置，不自动进入200步、不自动恢复中断训练。

## 输出和检查

- `manifest.json`：来源文件及源码哈希、50个训练样本及6个验证样本、完整参数、有效调度和运行环境。
- `summary.json`：运行状态、完成步数、配对检查、各验证节点的`pooled`汇总、`scenes`逐场景及`batches`逐batch指标。
- 每组`train/step_XXXX.json`：更新前与同batch更新后的损失/提案指标、差值、逐层梯度及更新方向、继承动量贡献、主干自初始化不变检查。
- 每组`step_0000/0001/0005/0010/0025.ckpt`：对应参数和优化器快照；另有实际optimizer_groups、mapping及模块模式记录。
- 每组`fingerprints/`：验证前向、训练前后、梯度、参数/优化器状态和训练入口随机状态的指纹。

复用单步诊断，每次真实更新后额外执行一次同batch无梯度前向，使用更新前BN/RNG入口；结束后恢复真实更新产生的状态。因而25步包含25次额外测量前向，不能按普通25步训练估算开销。

验证全模型eval，完成后恢复BN、模块模式和Python/NumPy/PyTorch随机状态，并检查模型/优化器及随机状态未被验证改变。验证不会增加训练BN计数。

keep_repeat完整轨迹必须exact；B组初始化、step0验证、首次训练前向和梯度必须exact，此后允许SGER差异，受保护raw/置信度、输入、主干状态和随机状态始终不得出现差异。残差上界、有限性、优化器覆盖、SGER更新及主干梯度隔离沿用已测试诊断检查。

汇总质量指标按stage3有效像素数加权；precision/recall先合并计数再计算，分母为零保留null。区域门控均值按各自改善/恶化像素数加权。残差目标/门控/完整loss的汇总是batch标量按stage3像素数加权的诊断值，不宣称等同于将全验证集一次性送入完整损失函数。原始batch标量全部保留。

## 判读及下一阶段

先检查`status=completed`、`checks_passed=true`、`input_files_unchanged=true`及三组completed_steps=25。这仅是运行可靠性结论。

比较各组相对自身step0的验证目标误差、提案收益、方向、幅度和逐场景趋势，不能把不同训练batch的损失直接当作单调学习曲线。检查B组的收益是否持续，是否只是局部缩小幅度，以及是否仍有验证场景持续恶化。

若B组跨batch和验证均改善，再考虑扩展200步；仅训练batch改善不足以进入正式训练。若提案收益恢复但门控仍阻断最终细化，固定优化策略后另立BCE单变量实验。不得同时修改学习率、margin和残差范围制造改善。

## 本地验证

```bash
OMP_NUM_THREADS=1 PYTHONPATH=.:tests python -m unittest discover -s tests -p 'test_momentum_probe.py'
```

测试采用生产CascadeMVSNet和损失的32×32 CPU合成样本，覆盖三组完整25步、验证节点、一次性清零及后续动量积累、样本偏移、重复精确一致、BN训练计数、主干保护、验证状态恢复、像素加权及计数聚合、原文件只读和样本清单篡改拒绝。同时复用单步诊断测试。真实数据只做预检，尚未执行本轮25步GPU实验。
