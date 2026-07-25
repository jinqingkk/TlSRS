# 实验10–12结果分析文档设计

## 目标

基于 `codex_results/实验结果.docx` 中记录的训练指标、DTU评价结果和点云数量，对实验10、实验11、实验12进行统一口径的结果对比与机制分析，并给出下一步 SGER-CasMVSNet 可直接执行和验证的网络优化方向。最终文档保存为 `codex_results/实验12分析.md`。

## 数据与实验边界

- 实验10：每个 cascade stage 后均运行 NormalHead 与 SGERBlock，refined depth 参与后续级联搜索。
- 实验11：Stage3-only SGER-Lite，最大残差比例为 0.5，包含 raw/refined/residual 联合监督和三种融合消融。
- 实验12：从零训练的 Stage3-only SGER-Lite；训练命令明确使用：
  - `sger_max_residual_ratio=0.25`
  - `raw_depth_loss_weight=1.0`
  - `refined_depth_loss_weight=1.0`
  - `residual_loss_weight=0.01`
  - `gate_loss_weight=0.001`
  - `safe_refine_loss_weight=0.1`
  - `safe_refine_margin=0`
  - `freeze_backbone_epochs=0`
  - `backbone_lr_scale=1.0`
- 实验12以 Word 文档中记录的数据为唯一结果来源。对话中检查到的另一轮全阶段 checkpoint 没有记录在 Word 文档中，不纳入分析。
- DTU 的 `acc`、`comp`、`overall` 均以越低越好解释。
- 对实验11使用 Epoch 12/14，对实验12使用 Epoch 12/15；跨实验比较时明确 epoch 不完全一致，重点比较稳定趋势和各实验自己的最佳记录。

## 分析结构

最终文档按“结果—机制—决策”组织：

1. 配置回顾和可比性边界。
2. 训练指标对比，突出深度收敛、normal-depth 一致性、gate 和 residual。
3. DTU结果对比，区分 raw backbone、refined depth 和 calibrated confidence 的贡献。
4. 点云数量对比，重点检查完整性损失和 Scan49 异常。
5. 实验10、11、12逐项结论。
6. 综合机制判断。
7. 下一步网络优化方向，按优先级给出具体结构、损失和参数。
8. 下一轮最小消融矩阵和固定评估协议。

## 关键判定规则

- 不以训练 `depth_loss` 单独判定模型优劣，最终以 DTU overall 为主。
- refined 的价值通过相同 checkpoint 下 refined 与 raw 的差值判断。
- calibrated confidence 的价值必须同时检查 acc、comp、overall 和点云数量。
- gate loss 是否有效通过 `geometry_gate_mean` 的动态和 refined 相对 raw 的结果共同判断。
- 实验12 Epoch 0–6 的异常不能归因于冻结主干，因为实际 `freeze_backbone_epochs=0`；应作为从零联合训练早期优化失稳分析。
- 若文档未记录某个指标，例如实验12 mean absolute residual 或 refined + raw confidence，则明确标注证据缺口，不以推测代替结果。

## 预期核心结论

- 实验10说明全阶段 SGER 和跨阶段 refined feedback 会显著破坏原有级联深度估计。
- 实验11说明 Stage3-only 能保护 raw backbone，但现有 refinement 和强置信度校准没有稳定改善最终结果。
- 实验12的 raw/raw 结果相对实验11 raw/raw 在 acc 上明显改善，但 comp 变差，overall 仍未超过实验11最佳；其 refined/calibrated 结果也没有超过 raw/raw。
- 实验12 gate 初期降到目标范围，后期却重新升至约 0.75，说明简单的 `mean(gate)` 正则不足以控制长期门控行为，或其有效权重被其他损失压制。
- 下一步不应扩大 SGER 容量或恢复多阶段 refinement，而应先修复优化稳定性、重新设计有目标占空比的 gate、让 residual 学习成为带置信判据的稀疏修正，并把 confidence calibration 与深度残差解耦。

## 输出质量要求

- 所有关键数值均可追溯到 Word 文档。
- 明确区分事实、推断和建议。
- 表格只保留支持决策的汇总数据，避免复制全部原始表。
- 网络优化建议必须包含修改位置、计算形式、推荐初值、观察指标和停止/继续条件。
- 文档中不引入未记录的全阶段实验12结果。
