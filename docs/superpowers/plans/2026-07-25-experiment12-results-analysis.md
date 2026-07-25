# 实验10–12结果分析文档实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从实验结果 Word 文档提取并核验实验10–12数据，生成包含结果对比、机制判断和下一步具体网络优化方案的 `codex_results/实验12分析.md`。

**Architecture:** 以 Word 文档为唯一结果数据源，将训练、DTU评价和点云数量压缩为决策型汇总表。分析严格区分记录事实、基于数据的推断和下一步建议，并将实验12限定为从零训练、Stage3-only、无冻结、等学习率配置。

**Tech Stack:** Markdown、DOCX Open XML、Python标准库、Git。

---

### Task 1: 核验实验10–12关键数据

**Files:**
- Read: `codex_results/实验结果.docx`
- Read: `codex_results/实验11分析.md`
- Read: `docs/superpowers/specs/2026-07-25-experiment12-results-analysis-design.md`

- [ ] **Step 1: 核验训练指标**

从 Word 表30、33、40记录实验10、11、12的 `depth_loss`、`abs_depth_error`、`normal_depth_cos`、`geometry_gate_mean` 和已提供的 residual 指标。实验12需保留 Epoch 0–6 异常和 Epoch 9–15恢复两段。

- [ ] **Step 2: 核验DTU汇总**

从 Word 表31–44记录实验10 Epoch 12/15、实验11 a/b/c Epoch 12/14、实验12 b/c Epoch 12/15的平均 `acc`、`comp`、`overall`。全部按越低越好解释。

- [ ] **Step 3: 核验点云数量**

从 Word 表45–46记录 baseline、实验10、实验11 a/b/c、实验12 b/c的总点数，并计算相对 baseline 的变化率，保留到两位小数。

- [ ] **Step 4: 复算关键差值**

使用：

```text
绝对差 = candidate - reference
相对变化率 = (candidate - reference) / reference × 100%
```

复算实验12相对实验11、实验10和baseline的 overall、acc、comp与点数变化，避免直接复用表中无标签差值。

### Task 2: 撰写结果与机制分析

**Files:**
- Create: `codex_results/实验12分析.md`

- [ ] **Step 1: 写实验设置与边界**

明确实验10全阶段、实验11 Stage3-only、实验12从零 Stage3-only，并写明实验12：

```text
freeze_backbone_epochs=0
backbone_lr_scale=1.0
```

排除未记录的全阶段训练结果。

- [ ] **Step 2: 写训练指标对比**

包含实验10–12关键训练指标表，并分别解释：

```text
实验10：全阶段feedback与训练劣化
实验11：raw路径受保护但gate约0.72
实验12：Epoch 0–6失稳、Epoch 9后恢复、gate最终约0.75
```

- [ ] **Step 3: 写DTU和点云对比**

建立最佳/代表性结果汇总表，明确实验11-c是三组中最佳，实验12-c优于实验12-b但不优于实验11-c；结合点云数量解释 calibration 和 Scan49 对 comp 的影响。

- [ ] **Step 4: 写综合结论**

区分：

```text
raw backbone贡献
SGER refined depth贡献
confidence calibration贡献
```

对未记录的实验12 refined+raw confidence和 residual 幅度明确标注证据缺口。

### Task 3: 给出下一步具体优化方向

**Files:**
- Modify: `codex_results/实验12分析.md`

- [ ] **Step 1: 给出P0优化**

建议先解决从零联合训练稳定性：SGER分支零残差启动、gate低占空比初始化、SGER损失warm-up，给出初值和观测条件。

- [ ] **Step 2: 给出P1优化**

将无目标的 `mean(gate)` 改为目标占空比/预算约束，并加入 gate 与“refined真实改善”之间的监督或停止梯度判据。

- [ ] **Step 3: 给出P2优化**

把 residual confidence calibration 从“残差越大越不可信”改为基于预测改善/不确定性的校准，默认评估仍保留 raw confidence。

- [ ] **Step 4: 给出最小消融矩阵**

下一轮按单变量顺序执行：

```text
A：稳定初始化
B：A + loss warm-up
C：B + gate target budget
D：C + improvement-aware residual
E：D + learned/validated confidence calibration
```

每组固定输出 raw+raw、refined+raw、refined+calibrated。

### Task 4: 文档验证与提交

**Files:**
- Verify: `codex_results/实验12分析.md`

- [ ] **Step 1: 检查必备章节**

运行：

```bash
rg -n "实验10|实验11|实验12|训练指标|DTU|点云|优化方向|消融|证据缺口" codex_results/实验12分析.md
```

预期所有主题均至少出现一次。

- [ ] **Step 2: 检查配置和关键数值**

运行：

```bash
rg -n "freeze_backbone_epochs=0|backbone_lr_scale=1.0|0.3543|0.3232|0.3264|0.3321|0.750" codex_results/实验12分析.md
```

预期所有配置和代表性数值均出现。

- [ ] **Step 3: 检查错误口径与占位符**

运行：

```bash
rg -n "越高越好|待补|未记录的全阶段实验12" codex_results/实验12分析.md
```

预期不出现“越高越好”或待补内容；允许以排除性说明出现未记录全阶段结果，但不得把它写入结果表。

- [ ] **Step 4: 检查Markdown和差异**

运行：

```bash
git diff --check -- codex_results/实验12分析.md
git diff -- codex_results/实验12分析.md
```

预期无空白错误，内容与设计规格一致。

- [ ] **Step 5: 提交文档**

```bash
git add codex_results/实验12分析.md
git commit -m "Analyze experiments 10 through 12"
```
