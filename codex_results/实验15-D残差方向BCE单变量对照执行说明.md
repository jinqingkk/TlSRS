# 实验15-D：残差方向BCE单变量对照

日期：2026-09-27。研究主体：**SGER-CasMVSNet（自几何与边缘细化残差-CasMVSNet网络）**。

## 问题、证据和唯一主要改动

固定验证集中，正残差目标占42.8437%，负残差目标占57.1529%，但已有各节点的残差提案仍全部为正。正负均衡Smooth L1在25步内主要将提案缩近零，没有形成有效负提案；门控归一化改动也只减轻了错误修正，没有使refined优于raw。

已有激活审计进一步显示，残差头`tanh`前输出及输出层权重贡献在已审计验证batch中均100%为正。输出bias约0.0024–0.0037，而输出层权重贡献均值约0.17，因此不能把全正提案简单归因于单个标量bias，也不优先采用清零bias作为修复。

本轮改善对象是SGER残差提案的方向表达。唯一主要变量是：在原生产总损失之外，增加对残差头`tanh`前logit的像素平均方向BCE。生产网络、生产损失默认配置和推理公式均不修改。独立入口为`experiment15_residual_direction_probe.py`。

## 三组与共同起点

| 组名 | 方向辅助监督 | 作用 |
|---|---|---|
| direction_reference | 无 | 原生产损失基准 |
| direction_repeat | 无 | 确定性重复 |
| sign_bce | 增加方向BCE，实际权重0.05 | 实验组 |

三组均恢复：

`outputs/experiment15_d/bce_pressure_25steps_epoch12_20260927_164443_461004/stable/step_0025.ckpt`

该起点已经执行25步稳定BCE压力更新，不是原始Epoch12 checkpoint。本轮从相同模型和完整Adam状态重新开始，不接续门控归一化实验，不清零动量，不使用scheduler。保存学习率、LeakyReLU(0.01)、稳定门控BCE、margin、残差范围及全部生产损失保持不变。

## 明确的优化边界

仅`SGERBlock.residual`残差头的8个参数张量可训练，包括三层body卷积及输出卷积的权重和bias。以下全部冻结：

- CasMVSNet raw主干及其优化器状态；
- 末级法线头、共享特征投影及其优化器状态；
- SGER收益门控A/B分支、几何区域和门控优化器状态；
- 全部BN缓冲区。

所有模块保持eval；eval不关闭残差头autograd。每步检查冻结模型参数、缓冲区及优化器状态完全一致，并验证raw深度、置信度、法线、benefit gate、门控logit、区域及几何权重不变。由此，组间残差提案及refined差异只能来自残差头更新和新增方向损失。

这一边界与此前“SGER及末级法线头共同训练”的实验不同，因此本轮只在三组内部归因。

## 方向标签、损失和梯度

使用生产`bounded_residual_target`构造区间限幅目标：

- target > 0：正方向标签1；
- target < 0：负方向标签0；
- target == 0：不参与方向BCE。

在整个batch的stage3 GT有效、非零目标像素上计算：

`L_direction = mean(BCEWithLogits(residual_logit, direction_label))`

实验组：

`L_total = L_production + 0.05 × L_direction`

基准和重复组仍计算方向BCE用于共同诊断，但有效方向权重为0，不加入总损失。权重0.05固定记录，不根据结果搜索。GT和有界目标均detach，只用于训练监督与评估，不进入推理输入；方向损失不能更新raw、门控或法线分支。

每次训练前向通过临时forward hook捕获活动残差输出卷积的`tanh`前logit，hook随后移除，不改变模型state_dict。首步要求正、负目标同时存在，并使用`autograd.grad`独立核验：

- 正目标的logit梯度全部小于0，梯度下降推动logit增大；
- 负目标的logit梯度全部大于0，梯度下降推动logit减小；
- 无效及零目标位置方向BCE梯度为0；
- 梯度全部有限。

这与此前正负均衡Smooth L1不同：方向BCE明确监督跨越零点，但不能预先保证网络特征具有足够区分能力，也可能导致全负或过度翻转。

## 数据、节点和结果字段

重放压力实验原有50个训练样本；2步使用前4条，25步使用全部50条。固定验证仍为scan3、scan5、scan17各2个参考视角。训练和验证均使用eval模式，验证no_grad且恢复运行状态。

保存节点为0、1、2、5、10、25（不超过指定步数）。`summary.json`的每个验证节点包含：

- `quality`：raw/refined误差、未门控提案收益、门控、有效残差等统一像素加权指标；
- `signed`与`signed_scenes`：正/负/零目标计数、提案正负比例、方向准确率、条件误差与收益；
- `selection`与`selection_scenes`：提案有益/有害、A/B分支及gate区间交叉统计；oracle仍仅为离线诊断；
- `batches`：逐验证batch损失、方向标签计数及诊断。

训练逐步文件记录更新前后指标、方向BCE、生产损失、实际方向权重、首步独立logit梯度检查，以及残差参数梯度/更新范数和点积。重复组全轨迹必须一致；实验组首步前向除总loss外必须一致。来源快照和源码哈希在结束时复核。

## 执行顺序

可选CPU只读预检：

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. python -u experiment15_residual_direction_probe.py \
  --probe_dir outputs/experiment15_d/bce_pressure_25steps_epoch12_20260927_164443_461004 \
  --steps 25 --preflight_only
```

先运行真实2步审计：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 PYTHONPATH=. \
python -u experiment15_residual_direction_probe.py \
  --probe_dir outputs/experiment15_d/bce_pressure_25steps_epoch12_20260927_164443_461004 \
  --steps 2 --strict_determinism
```

确认completed、checks_passed、input_files_unchanged，重复组一致、首步方向梯度正确、冻结边界通过后，再独立运行25步：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 PYTHONPATH=. \
python -u experiment15_residual_direction_probe.py \
  --probe_dir outputs/experiment15_d/bce_pressure_25steps_epoch12_20260927_164443_461004 \
  --steps 25 --strict_determinism
```

25步重新从共同起点开始，不接续2步末尾。默认输出为`outputs/experiment15_d/residual_direction_2steps_时间戳`或`residual_direction_25steps_时间戳`；`--output_dir`只接受不存在的新目录。

## 判定标准

首先判断对照是否合法，然后按以下顺序判断方向恢复：

1. 负目标区域出现具有实际数量的负提案，负方向准确率提高；只出现极少数符号翻转不足以判定恢复。
2. 正目标区域仍主要产生正提案，不能从全正塌缩为全负。
3. 未门控提案平均收益转正，且scan3、scan5、scan17方向一致；仅缩小提案接近0不算成功。
4. 在固定门控下，refined实际优于raw；同时报告门控可能限制已改善提案的程度。
5. raw、门控及所有冻结状态必须保持一致，不能通过损害对照或改变门控制造相对优势。

若方向BCE仍只缩小整体幅度、不能跨零，或产生全负塌缩，则否定当前辅助监督假设，再考虑方向×幅度因子化残差头。不得同时改变方向损失权重、残差结构、学习率和门控策略后归因于其中一项。

本轮是有限样本短程机制对照，不代替正式训练与同raw confidence、同融合协议的DTU Acc./Comp./Overall评估。

## 本地验证状态

```bash
OMP_NUM_THREADS=1 PYTHONPATH=.:tests python -m unittest discover -s tests -p 'test_residual_direction_probe.py'
```

4项测试全部通过（100.714秒），包括方向BCE公式与梯度、GT/raw detach、零目标排除、极端logit有限性、错误梯度拒绝，以及生产网络CPU合成数据的2步和25步三组完整对照。测试确认只有残差头8个张量改变，固定raw与门控保持一致，重复组轨迹一致，实验组首步梯度不同。

真实来源CPU只读预检通过：恢复快照为`stable/step_0025.ckpt`，仅8个残差头参数张量可训练，25步将重放原50个样本。真实GPU审计由用户自行执行，本次尚无实际精度结果。

