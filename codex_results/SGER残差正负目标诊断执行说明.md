# SGER残差正负目标诊断执行说明

日期：2026-09-21。研究主体：**SGER-CasMVSNet（自几何与边缘细化残差-CasMVSNet网络）**。

25步动量对照未恢复有效双向纠错。本轮新增`experiment15_residual_sign_audit.py`，复用已有keep与zero_residual_exp_avg第0、10、25步共6份快照，只做前向和autograd.grad，不执行optimizer.step、不改写快照或生产网络。

## 固定样本和口径

训练子集选原25步清单中第1、13、25个batch（零基位置0、12、24），共6个样本；验证仍使用原scan3、scan5、scan17各两张参考图。选择不依赖误差结果。两类数据集均使用model.eval()，从而固定网络运行模式；数据加载保留原train/val模式和选择元数据。

stage3采用GT mask>0.5，目标调用生产`bounded_residual_target`，严格按目标>0、<0、==0分组，不使用额外epsilon。零目标组不算方向正确，空组统计为null。

每组统计像素数、占比、提案正/负/零比例、提案幅度、目标误差、raw/refined误差、未门控提案收益和门控均值。汇总按各组有效像素数加权，不能直接平均不同batch的百分比。

## 梯度解释

对每个非空符号组，按生产参数化计算组内平均smooth L1残差目标损失，记录残差头逐参数梯度及d_loss/d_proposal。另乘以“实际残差目标权重×组像素数/全有效像素数”，得到该组在完整残差目标损失中的加权贡献。正负组的梯度余弦仅描述残差头上的方向关系，不涵盖其他损失或共享分支梯度冲突。

`mean_proposal_descent_derivative`是在残差头参数上沿该组加权负梯度作无穷小SGD扰动时，组内平均提案的一阶变化率。负目标组出现负值，说明局部梯度有推动提案下降的能力；不代表真实Adam会采用该方向，也不证明有限步长后提案已经跨零。程序不执行虚拟参数更新。

逐batch保留激活及bias-only差异、梯度报告；汇总不对参数梯度范数作像素加权，以免混淆梯度相加和范数相加。完整损失优化冲突及实际跨零能力需根据结果另行验证。

## 命令

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 PYTHONPATH=. python -u experiment15_residual_sign_audit.py \
  --probe_dir outputs/experiment15_residual_recovery/momentum_25steps_epoch12_20260921_204158_936438 \
  --strict_determinism
```

可选CPU只读预检：

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. python -u experiment15_residual_sign_audit.py \
  --probe_dir outputs/experiment15_residual_recovery/momentum_25steps_epoch12_20260921_204158_936438 \
  --preflight_only
```

默认保存至`outputs/experiment15_residual_recovery/sign_audit_时间戳/`。新目录内`summary.json`汇总两组、三个节点、train/val的正负零目标统计，各组目录下的`step*_train/val_batch*.json`包含分组梯度与激活详情。

先检查status=completed、checks_passed=true、input_files_unchanged=true；然后比较负目标占比、负目标区正提案比例、负目标误差，以及负组对输出偏置/权重的梯度。若负目标梯度正常却提案始终为正，重点研究输出层符号表达和实际优化响应；若负组梯度为零或没有连接，再定位梯度路径。不要据单个统计直接改动全部训练配置。

## 保护和验证

核验原25步实验完成状态、源码哈希、快照参数/优化器指纹及数据元信息；第0步B组只允许已声明的一阶动量清零差异。跨所有快照检查相同样本的输入、raw和原置信度一致，结束检查模型/优化器状态不变、参数.grad仍为None、来源文件哈希不变。所有输出写入新目录。

本地2项测试已通过：带正负零目标的解析例子验证梯度符号、组损失分解及空组语义；生产CascadeMVSNet的CPU合成例子验证eval前后状态不变、梯度仅返回且不累积、无optimizer更新。6份真实快照只读预检通过，尚未执行真实DTU/GPU诊断。

```bash
OMP_NUM_THREADS=1 PYTHONPATH=.:tests python -m unittest discover -s tests -p 'test_residual_sign_audit.py'
```

本轮是有限固定样本的机制诊断，不是正式训练，也不提供DTU Acc./Comp./Overall结论。
