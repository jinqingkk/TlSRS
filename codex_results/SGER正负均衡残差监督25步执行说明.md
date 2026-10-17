# SGER正负均衡残差监督25步执行说明

日期：2026-09-21。研究主体：**SGER-CasMVSNet（自几何与边缘细化残差-CasMVSNet网络）**。

新增独立入口`experiment15_sign_balanced_probe.py`，检验训练中残差正负目标的权重竞争是否影响SGER双向纠错。实验不修改生产网络或已有损失函数，所有恢复与单步/25步历史实验入口保持原样。

## 唯一主要变量与起点

三组均恢复原两步实验的`leaky_relu/step_0001.ckpt`，这是此前25步动量实验共同的step0起点。不要加载动量实验第10/25步或清零组末尾快照。

| 目录 | 残差目标监督 | 优化器 |
|---|---|---|
| pixel_mean | 原有效像素平均smooth L1 | 完整恢复Adam |
| pixel_repeat | 与pixel_mean相同 | 重复对照 |
| sign_balanced | 下述正负组均衡smooth L1 | 完整恢复Adam，不清零动量 |

固定LeakyReLU(0.01)、legacy_clamp、主干冻结和BN eval、残差限幅、收益margin、全部其他损失及实际权重，沿用Epoch12调度与恢复学习率，无scheduler。SGER及末级法线头保持可训练。GT只用于训练损失和评估，不进入网络推理输入。

## 均衡公式和缺失组处理

在当前batch的stage3 GT有效掩码上计算生产有界残差目标。严格按目标>0、<0、==0分组，计数分别为N+、N−、N0，总数N；组内平均smooth L1分别为L+、L−、L0。

正负组都存在时：

`L_balanced = (N+ + N−)/(2N) × (L+ + L−) + (1/N) × sum(zero组逐像素损失)`

即非零像素总权重保持不变，正负两组各分一半；零目标仍保持原每像素1/N权重。正负任一组缺失时直接使用原像素平均损失；空有效掩码视为错误。归一化在整个batch内进行，不按单张图分别均衡。

实验组用`production_total + effective_weight × (L_balanced − L_pixel)`替换残差目标项，其余生产损失不变。本轮effective_weight=0.05，来自实际调度而非写死。替换前逐次核验重算L_pixel与生产残差目标损失一致，防止单位或归一化不一致。

逐batch记录三类计数、是否启用均衡、正负/零每像素权重倍率、原损失、均衡损失和实际生效损失。该策略不保证产生负提案，有可能仅缩小整体残差，因此不能以损失下降单独判定成功。

## 数据和评估

复用原样本清单[2:52]的50个训练样本和原6个验证样本，不因诊断结果重新挑选。训练25步，验证/快照节点0、1、5、10、25。local step0是恢复原第1步快照后的状态。

验证全模型eval，统一按pixel_mean口径计算完整loss及目标loss，B组也不更改验证定义；额外保留balanced_target_loss作诊断。验证不更新参数，并恢复模式、BN缓冲区和随机状态。

汇总保留原像素加权质量指标、计数汇总后的门控precision/recall，另增`signed`和`signed_scenes`：正负零目标的数量、比例、提案符号、方向准确率、目标误差、提案收益和raw/refined误差。空组为null，零目标不计为方向正确。逐场景和逐batch记录均保留。

每个训练步继续测量同batch更新前后指标，并记录各参数的梯度/实际更新方向；测量前向的BN/RNG按原单步诊断隔离，因此每步需要额外一次无梯度前向。

## 执行

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 PYTHONPATH=. python -u experiment15_sign_balanced_probe.py \
  --probe_dir outputs/experiment15_residual_recovery/probe_2steps_epoch12_20260920_214234_152286 \
  --strict_determinism
```

注意：这里的--probe_dir指向**原两步实验目录**，不是正负诊断或25步动量输出目录。

可选CPU只读预检：

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. python -u experiment15_sign_balanced_probe.py \
  --probe_dir outputs/experiment15_residual_recovery/probe_2steps_epoch12_20260920_214234_152286 \
  --preflight_only
```

输出为`outputs/experiment15_residual_recovery/sign_balanced_25steps_epoch12_时间戳/`。`--output_dir`只接受新目录。原快照和历史输出只读，不自动进入200步或正式训练。

## 配对和结果判定

重复组全部轨迹精确一致；B组初始化与step0验证必须一致，首次训练前向除loss之外的全部输入/输出必须一致。由于监督变化，首次梯度允许不同。后续始终检查raw/置信度、输入、随机状态和主干参数/BN/优化器状态保护。

先看summary的完成状态、检查标志、原文件不变及三组25步完成情况，再看：

1. 验证负目标区域是否出现负提案，其比例和条件误差是否改善。
2. 正目标区域是否保持有效修正，避免从全正塌缩到全负。
3. 总体验证未门控提案收益是否转正且逐场景一致；仅缩小幅度接近零不算双向纠错恢复。
4. refined是否实际优于raw；门控贡献仍需独立判断。

不要跨不同损失定义直接比较训练总loss选优；始终用相同验证质量指标比较。这是有限样本机制对照，不替代DTU Acc./Comp./Overall验收。

## 本地验证状态

```bash
OMP_NUM_THREADS=1 PYTHONPATH=.:tests python -m unittest discover -s tests -p 'test_sign_balanced_probe.py'
```

5项测试通过，包括生产网络CPU合成数据的三组完整25步，合成GT强制包含不均衡正负目标，确认均衡实际激活、首步梯度不同而raw受保护。另验证公式及梯度替换边界、GT detach、零目标权重、缺失符号回退、无动量清零、BN计数、重复轨迹、签名分组汇总、样本清单保护，并复用已有单步诊断测试。真实快照和样本清单预检通过，尚未运行真实GPU对照。
