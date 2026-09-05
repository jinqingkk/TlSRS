# SGER-CasMVSNet 网络架构与进一步研究方案

## 1. 网络名称与研究定位

SGER-CasMVSNet 的全称为：

> **Self-Geometric Edge-aware Refinement Cascade Multi-View Stereo Network**

中文可表述为：

> **自几何约束与边缘感知细化级联多视图立体网络**

其中：

- **SG（Self-Geometric）**：利用网络自身产生的深度、预测法线、深度反算法线、概率置信度、深度梯度和曲率构造几何信息，不依赖额外真值法线；
- **E（Edge-aware）**：利用参考图像边缘和深度边缘区分连续表面与结构边界，控制细化位置和强度；
- **R（Refinement）**：在每个级联阶段加入可学习的残差深度细化模块，并将细化深度反馈到下一阶段；
- **CasMVSNet**：保留原始三级粗到细特征、深度采样、单应重投影、方差代价体和 3D 代价正则化主体。

SGER-CasMVSNet 是在当前“CasMVSNet + 逐阶段法线分支 + 几何训练损失”基础上的进一步研究方案。本文所述可学习 SGER 模块属于**拟议新增**，目前**不在当前实际前向路径中**。因此，下文把“当前已实现”与“拟议新增”明确区分，避免将研究假设误写成实验结论。

## 2. 研究动机

当前改进网络已经能够利用深度—法线一致性、曲率连续性、边缘感知平滑和 Region A/B 双区域机制改善训练监督，但这些几何信息主要通过损失函数间接作用于网络参数。推理阶段仍然由 Stage3 深度回归直接给出最终结果，没有一个显式利用几何与边缘信息修正深度的可学习模块。

进一步研究的核心问题是：

> 能否把训练阶段有效的自几何信息和边缘调制机制转化为推理阶段可执行的深度细化模块，并通过跨阶段反馈改善后续深度假设采样？

该研究针对以下现象提出：

1. 低纹理区域的概率体可能较平坦，深度期望容易漂移；
2. 大面积混凝土弧面需要稳定的二阶几何连续性；
3. 拼接缝、孔洞、遮挡和轮廓附近不应被统一平滑；
4. Stage1 或 Stage2 的误差会改变下一阶段采样中心，造成级联误差传播；
5. 当前法线和几何信息在推理时没有直接参与深度修正。

SGER 的目标不是替代代价体，而是在每阶段深度回归后，对局部残差进行受控修正。

## 3. 当前已实现基线与拟议新增内容

| 组成 | 当前状态 | SGER-CasMVSNet 中的作用 |
|---|---|---|
| FPN 三尺度特征 | 当前已实现 | 保持不变 |
| 三级深度范围采样 | 当前已实现 | Stage2/3 改用细化深度作为采样中心 |
| 单应重投影与方差代价体 | 当前已实现 | 保持不变 |
| 分阶段 `CostRegNet` | 当前已实现 | 保持不变 |
| 深度、置信度回归 | 当前已实现 | 产生 SGER 的基础输入 |
| 逐阶段 `NormalHead` | 当前已实现 | 提供预测法线 |
| 几何/边缘损失 | 当前已实现 | 迁移到细化深度并继续使用 |
| `GeometryCueExtractor` | 拟议新增 | 计算深度法线、边缘、深度梯度和曲率 |
| `DualRegionGate` | 拟议新增 | 生成 Region A/B 调制门控 |
| `ResidualDepthHead` | 拟议新增 | 预测受限深度残差 |
| `SGERBlock` | 拟议新增 | 组合几何特征、门控和残差细化 |
| 细化深度跨阶段反馈 | 拟议新增 | 用 $\widetilde D_s$ 驱动下一阶段采样 |

## 4. SGER-CasMVSNet 总体架构

### 4.1 主数据流

```text
多视图图像与相机参数
        │
        ▼
    FeatureNet / FPN
        │
        ├──────────────────────────────────────────────┐
        ▼                                              │
Stage1: Cost Volume → D1, C1 → NormalHead1 → N1        │
        │                    │                         │
        └───────────── SGER1(D1,N1,C1,Iref,K1,F1)      │
                              │                         │
                              ▼                         │
                            D̃1                         │
                              │ 作为局部采样中心         │
                              ▼                         │
Stage2: Cost Volume → D2, C2 → NormalHead2 → N2        │
        │                    │                         │
        └───────────── SGER2(D2,N2,C2,Iref,K2,F2)      │
                              │                         │
                              ▼                         │
                            D̃2                         │
                              │ 作为局部采样中心         │
                              ▼                         │
Stage3: Cost Volume → D3, C3 → NormalHead3 → N3        │
                             │                          │
                      SGER3(D3,N3,C3,Iref,K3,F3)       │
                             │                          │
                             ▼                          │
                    最终细化深度 D̃3                    │
```

### 4.2 阶段输出定义

每个原始 CasMVSNet 阶段先产生：

$$
(D_s,C_s,N_s^{\mathrm{pred}})
=\operatorname{Stage}_s
(F_s,P_s,\mathcal D_s),
$$

其中：

- $D_s$：原始阶段深度；
- $C_s$：概率体光度置信度；
- $N_s^{\mathrm{pred}}$：网络预测法线；
- $F_s$：参考视图阶段特征；
- $P_s$：阶段相机参数；
- $\mathcal D_s$：阶段深度假设。

SGER 输出：

$$
\widetilde D_s=
\operatorname{SGER}_s
(D_s,N_s^{\mathrm{pred}},C_s,I_{\mathrm{ref},s},K_s,F_s).
$$

Stage1 和 Stage2 的 $\widetilde D_s$ 取代原始 $D_s$ 成为下一阶段深度采样中心；Stage3 的 $\widetilde D_3$ 为最终深度。

## 5. 每阶段 SGERBlock 的输入与输出

### 5.1 必需输入

| 输入 | 形状 | 来源 | 作用 |
|---|---|---|---|
| $D_s$ | $(B,1,H_s,W_s)$ | 深度回归 | 原始深度和残差基准 |
| $N_s^{pred}$ | $(B,3,H_s,W_s)$ | `NormalHead` | 网络学习的表面方向 |
| $C_s$ | $(B,1,H_s,W_s)$ | 概率体 | 可靠性调制 |
| $I_{ref,s}$ | $(B,3,H_s,W_s)$ | 缩放参考图像 | 图像边缘与外观结构 |
| $K_s$ | $(B,3,3)$ | 阶段相机内参 | 深度反算法线 |

### 5.2 可选输入

参考特征 $F_s$ 可先通过 $1\times1$ 卷积投影为固定通道数 $C_g$，再输入残差网络：

$$
F_s^g=\operatorname{Conv}_{1\times1}(F_s).
$$

该输入提供局部语义和纹理信息。为了判断 SGER 的收益来自几何门控还是高维特征，研究中应设置“不使用 $F_s$”的消融组。

### 5.3 输出

SGER 至少输出：

```python
{
    "depth_refined": D_tilde_s,
    "depth_residual": Delta_D_s,
    "geometry_gate": G_s,
    "region_a": R_A_s,
    "region_b_weight": W_B_s,
}
```

训练时保留中间输出用于损失和可视化；推理时可只保留最终细化深度和必要诊断图。

## 6. GeometryCueExtractor：自几何信息提取

### 6.1 相机感知深度法线

由原始预测深度恢复相机坐标点：

$$
X_s(u,v)=D_s(u,v)K_s^{-1}[u,v,1]^\mathsf T.
$$

利用相邻三维点构造切向量：

$$
T_x=X_s(u+1,v)-X_s(u,v),
$$

$$
T_y=X_s(u,v+1)-X_s(u,v).
$$

深度反算法线为：

$$
N_s^D=\operatorname{normalize}(T_x\times T_y).
$$

### 6.2 法线不一致度

预测法线和深度法线的差异可直接作为细化线索：

$$
R_s^N=1-
\left|
N_s^{pred}\cdot N_s^D
\right|.
$$

$R_s^N$ 较大表示当前深度形状和网络法线判断不一致，但这种不一致也可能来自法线预测噪声，因此不能单独决定残差。

### 6.3 归一化深度梯度

$$
D_s^n=\frac{D_s}
{\operatorname{Mean}_{M_s}(D_s)+\varepsilon},
$$

$$
G_s^D=\max
\left(
|\partial_xD_s^n|,
|\partial_yD_s^n|
\right).
$$

### 6.4 深度曲率

$$
\kappa_s=max
\left(
|\Delta_{xx}D_s^n|,
|\Delta_{yy}D_s^n|
\right).
$$

### 6.5 几何特征集合

自几何分支输出：

$$
\Phi_s^G=
\left[
D_s^n,
N_s^{pred},
N_s^D,
R_s^N,
C_s,
G_s^D,
\kappa_s
\right].
$$

除 $N_s^{pred}$、$D_s$ 和后续残差路径外，用于阈值划分的深度、置信度和派生区域建议使用分离副本，避免离散门控产生不稳定梯度。

## 7. EdgeCueExtractor：边缘感知信息

### 7.1 Sobel 图像边缘

将参考图像转换为灰度图 $I_g$，计算：

$$
E_s^I=
\sqrt{(S_x*I_g)^2+(S_y*I_g)^2+\varepsilon}.
$$

### 7.2 图像边缘与深度边缘关系

单独的图像边缘不一定是几何边界，例如纹理、污渍和光照变化也会产生强边缘。因此，结构区域应综合图像边缘和深度边缘，而不是仅由 $E_s^I$ 决定。

边缘分支输出：

$$
\Phi_s^E=
\left[I_{ref,s},E_s^I,G_s^D\right].
$$

## 8. DualRegionGate：双区域几何门控

### 8.1 Region A：结构或高曲率区域

$$
R_s^A=M_s\land
\left[
(E_s^I>\tau_I\land G_s^D>\tau_D)
\lor(\kappa_s>\tau_\kappa)
\right].
$$

Region A 包含联合图像—深度边缘和高曲率区域。它采用硬区域支持，但区域内部的残差幅度仍由可学习门控决定：

$$
g_s^A=\sigma\left(h_A(Z_s)\right).
$$

### 8.2 Region B：连续表面区域

$$
R_s^B=M_s\land\neg R_s^A.
$$

先构造置信度与边缘软权重：

$$
w_s^B=\mathbf1_{R_s^B}
\sigma\left(k_c(C_s-c_0)\right)
\exp(-k_eE_s^I).
$$

再由可学习分支产生细化门控：

$$
g_s^B=w_s^B\odot
\sigma\left(h_B(Z_s)\right).
$$

### 8.3 融合门控

$$
G_s=
\mathbf1_{R_s^A}\odot g_s^A
+\mathbf1_{R_s^B}\odot g_s^B.
$$

其中 $Z_s$ 是几何、边缘、参考特征的融合张量。Region A/B 的二值构造建议基于 `detach()` 后的深度和置信度；$h_A$、$h_B$ 及后续残差网络保持可导。

## 9. ResidualDepthHead：受控残差深度预测

### 9.1 特征融合

残差网络输入可定义为：

$$
Z_s=\operatorname{Concat}
\left(
\Phi_s^G,
\Phi_s^E,
F_s^g
\right).
$$

一个轻量实现可以使用：

```text
3×3 Conv → 32 channels
3×3 Dilated Conv(dilation=2) → 32 channels
3×3 Conv → 16 channels
3×3 Conv → 1 residual logit
```

可加入残差块或深度可分离卷积，但第一版应保持轻量，以便明确收益是否来自几何机制而不是显著增加模型容量。

### 9.2 残差幅度限制

为防止细化结果跳出当前阶段的合理深度范围，使用 `tanh` 限制残差：

$$
\Delta D_s=
\alpha_s\delta_s
\tanh\left(h_\Delta(Z_s)\right),
$$

其中 $\delta_s$ 是当前阶段基础深度间隔，$\alpha_s$ 是最大残差比例。建议初始实验采用 $\alpha_s\in[1,2]$，具体值需要通过验证集选择。

### 9.3 门控残差细化

最终细化公式为：

$$
\boxed{
\widetilde D_s=D_s+G_s\odot\Delta D_s
}
$$

门控控制“在哪里修正”，残差网络控制“修正方向和幅度”。

## 10. 跨阶段反馈

### 10.1 反馈方式

当前 CasMVSNet 使用上一阶段原始深度作为下一阶段采样中心。SGER-CasMVSNet 改为：

$$
D_{s\rightarrow s+1}^{center}
=\operatorname{Resize}(\widetilde D_s).
$$

局部深度范围为：

$$
d_{s+1,k}(\mathbf p)=
\widetilde D_s(\mathbf p)
+\left(k-\frac{D_{s+1}^{\#}-1}{2}\right)
\Delta d_{s+1}(\mathbf p).
$$

### 10.2 反馈风险

若早期 SGER 输出错误，后续深度范围可能围绕错误中心收窄，产生级联放大。因此建议提供以下训练选项：

- `detach_refined_feedback=True`：默认分离细化深度后再作为下一阶段中心；
- 训练早期使用原始 $D_s$ 作为中心，稳定后切换到 $\widetilde D_s$；
- 设置残差上限，避免细化中心大幅偏移；
- 记录细化前后深度误差，确认 SGER 在各阶段实际改善而非仅改善 Stage3。

## 11. 完整训练目标

### 11.1 原始深度辅助监督

为保持原 CasMVSNet 主干稳定，保留：

$$
\mathcal L_{raw}^s=
\operatorname{SmoothL1}(D_s,D_s^{gt}).
$$

### 11.2 细化深度主监督

$$
\mathcal L_{ref}^s=
\operatorname{SmoothL1}(\widetilde D_s,D_s^{gt}).
$$

### 11.3 深度—法线一致性

使用细化深度反算几何法线：

$$
\widetilde N_s^D=
\operatorname{NormalFromDepth}(\widetilde D_s,K_s),
$$

$$
\mathcal L_{dn}^s=
\operatorname{Mean}_{M_s^{smooth}}
\left[
1-|N_s^{pred}\cdot\widetilde N_s^D|
\right].
$$

### 11.4 法线平滑、曲率与边缘感知平滑

将现有损失的深度输入从原始 $D_s$ 替换或扩展为细化深度 $\widetilde D_s$：

$$
\mathcal L_{ns}^s=
\operatorname{NormalSmooth}(\widetilde D_s,M_s),
$$

$$
\mathcal L_{curv}^s=
\lambda_A\mathcal L_A(\widetilde D_s)
+\lambda_B\mathcal L_B(\widetilde D_s,w_s^B),
$$

$$
\mathcal L_{edge}^s=
\operatorname{EdgeAwareSmooth}
(\widetilde D_s,I_{ref,s},M_s).
$$

### 11.5 残差幅度正则

$$
\mathcal L_{res}^s=
\operatorname{Mean}_{M_s}
\left(
\frac{|G_s\odot\Delta D_s|}
{\delta_s+\varepsilon}
\right).
$$

该项抑制不必要的大幅修改，使 SGER 更接近局部细化而不是重新预测深度。

### 11.6 总损失

$$
\begin{aligned}
\mathcal L_{total}=\sum_{s=1}^{3}\omega_s\Big[&
\eta_{raw}\mathcal L_{raw}^s
+\eta_{ref}\mathcal L_{ref}^s
+\lambda_{dn}\mathcal L_{dn}^s\\
&+\lambda_{ns}\mathcal L_{ns}^s
+\lambda_{curv}\mathcal L_{curv}^s
+\lambda_{edge}\mathcal L_{edge}^s
+\lambda_{res}\mathcal L_{res}^s
\Big].
\end{aligned}
$$

建议初始实验采用：

- $\eta_{raw}=0.5$，保留主干辅助监督；
- $\eta_{ref}=1.0$，以细化深度为主要目标；
- 几何损失先沿用当前数量级：$\lambda_{dn}=0.03$、$\lambda_{ns}=0.02$、$\lambda_{curv}=0.005$、$\lambda_{edge}=0.005$；
- $\lambda_{res}$ 从 $10^{-3}$ 附近开始搜索；
- 以上仅为实验起点，不是已验证的最优参数。

## 12. 梯度传播设计

| 路径 | 建议梯度策略 | 原因 |
|---|---|---|
| $D_s\rightarrow\Delta D_s\rightarrow\widetilde D_s$ | 保留梯度 | 训练残差细化并更新深度分支 |
| $N_s^{pred}\rightarrow$ SGER 特征 | 保留梯度 | 让法线分支参与细化学习 |
| $C_s\rightarrow$ 连续可学习特征 | 可保留或分离，作为消融 | 置信度当前在 `no_grad` 中生成 |
| $D_s,C_s\rightarrow R_s^A,R_s^B$ | 分离 | 避免阈值区域产生不稳定梯度 |
| $G_s,\Delta D_s\rightarrow\widetilde D_s$ | 保留梯度 | 训练门控和残差网络 |
| $\widetilde D_s\rightarrow$ 下一阶段采样 | 默认分离 | 降低跨阶段反馈不稳定性 |

若后续要让置信度参与端到端门控学习，需要把当前 `DepthNet` 中置信度计算移出 `torch.no_grad()`，并单独研究其数值稳定性。

## 13. 训练策略

### 13.1 阶段一：基线预训练或加载

先训练当前“CasMVSNet + NormalHead + 几何损失”，或加载其稳定检查点，使深度、置信度和法线具有合理初值。

### 13.2 阶段二：SGER 冷启动

- 将 `ResidualDepthHead` 最后一层权重和偏置初始化为零；
- 使初始 $\Delta D_s\approx0$，因此 $\widetilde D_s\approx D_s$；
- 暂时使用原始 $D_s$ 驱动下一阶段采样；
- 冻结或降低 CasMVSNet 主干学习率，只训练 SGER 若干轮。

### 13.3 阶段三：启用跨阶段反馈

- 切换为 $\widetilde D_1$、$\widetilde D_2$ 驱动后续采样；
- 默认对反馈深度执行 `detach()`；
- 联合微调主干、NormalHead 和 SGER；
- 对梯度范数、残差幅度和采样范围越界率进行监控。

### 13.4 推荐监控指标

- 每阶段 `raw_depth_loss` 与 `refined_depth_loss`；
- $\operatorname{mean}|\Delta D_s|$ 和最大残差比例；
- `gate_mean`、`gate_high_ratio`；
- Region A/B 像素比例；
- 细化前后深度绝对误差差值；
- 深度—法线余弦一致性；
- 下一阶段局部范围覆盖真值深度的比例。

## 14. 推理流程与输出

推理时不计算损失，但每阶段执行 SGER：

```text
Stage1 raw depth → SGER1 → refined depth → Stage2 range
Stage2 raw depth → SGER2 → refined depth → Stage3 range
Stage3 raw depth → SGER3 → final refined depth
```

建议输出结构：

```python
outputs["stage1"] = {
    "depth_raw": D1,
    "depth_refined": D1_tilde,
    "depth": D1_tilde,
    "photometric_confidence": C1,
    "normal": N1,
    "geometry_gate": G1,
}

outputs["stage2"] = {...}
outputs["stage3"] = {...}

outputs["depth"] = D3_tilde
outputs["normal"] = N3
outputs["photometric_confidence"] = C3
```

保留 `depth_raw` 有利于评估 SGER 的净贡献；`depth` 继续表示当前阶段实际传递和使用的深度，便于兼容现有推理代码。

## 15. 建议的软件模块与代码集成

### 15.1 新增文件

建议新增 `models/sger_refinement.py`，避免继续扩大 `models/module.py`：

```python
class GeometryCueExtractor(nn.Module):
    """Compute depth normals, normal disagreement, depth edges and curvature."""

class DualRegionGate(nn.Module):
    """Build detached regions and differentiable hard/soft gated weights."""

class ResidualDepthHead(nn.Module):
    """Predict a bounded stage-wise depth residual."""

class SGERBlock(nn.Module):
    """Fuse cues, predict gate/residual and return refined depth plus diagnostics."""
```

### 15.2 修改 `CascadeMVSNet`

新增构造参数：

```text
use_sger
sger_share
sger_feature_channels
sger_max_residual_ratio
detach_refined_feedback
```

在每阶段 `NormalHead` 之后调用 `SGERBlock`，并将细化深度保存为下一阶段 `cur_depth`。

### 15.3 修改损失

新增或扩展：

```text
raw_depth_loss_weight
refined_depth_loss_weight
residual_loss_weight
geometry losses applied to depth_refined
```

### 15.4 修改训练参数和日志

`train.py` 增加 SGER 开关、残差上限、反馈分离和损失权重参数；TensorBoard 增加残差图、门控图、细化前后深度误差和区域统计。

### 15.5 检查点兼容

- `use_sger=False` 时必须与当前网络行为一致；
- 从旧检查点加载时允许缺少 SGER 参数，并对新增层执行确定性初始化；
- 保存检查点时记录 SGER 配置，防止推理参数不匹配。

## 16. 消融实验设计

| 实验组 | 目的 |
|---|---|
| 当前改进 CasMVSNet 基线 | 建立无可学习细化的参照 |
| 仅 Stage3 SGER | 判断单次最终细化的收益和成本 |
| 全阶段 SGER | 验证跨阶段反馈是否有效 |
| 仅 Residual CNN，无几何门控 | 判断普通 CNN 容量带来的收益 |
| 仅几何规则校正，无 Residual CNN | 判断规则本身的贡献 |
| Hard Region A only | 分析硬区域作用 |
| Soft Region B only | 分析连续表面软权重作用 |
| Dual Region A+B | 验证双区域互补性 |
| 共享 SGER 参数 | 评估参数效率和跨尺度泛化 |
| 独立 SGER 参数 | 评估尺度专用建模能力 |
| 去除 raw-depth 辅助监督 | 判断主干稳定性 |
| 去除残差正则 | 判断过度修正风险 |
| 分别去除 DN/曲率/边缘损失 | 分析各几何约束贡献 |

所有实验应保持训练数据、视图数、深度假设、训练轮数和融合阈值一致。

## 17. 评价指标

### 17.1 深度估计指标

- 平均绝对深度误差；
- RMSE；
- 不同阈值下的深度准确率；
- Stage1/2/3 细化前后误差变化；
- 真值深度落入下一阶段采样范围的覆盖率。

### 17.2 三维重建指标

- DTU Accuracy；
- DTU Completeness；
- Overall；
- 点云完整率和离群点比例。

### 17.3 几何与边界指标

- 预测法线与深度法线平均余弦；
- 非边缘区域深度梯度均值；
- 曲率误差或曲率方差；
- 深度边界 F1 或边界附近深度误差；
- Region A/B 分区内的独立深度误差。

### 17.4 效率指标

- 参数量；
- 单张/单场景推理时间；
- 峰值 GPU 显存；
- 每阶段 SGER 额外 FLOPs；
- 相对基线的速度下降比例。

## 18. 研究假设

以下内容是需要实验验证的假设，不是既定结果：

1. 全阶段 SGER 可能通过改善后续采样中心优于仅 Stage3 细化；
2. Region B 软门控可能提高大面积连续弧面的稳定性；
3. Region A 硬支持和可学习幅度可能兼顾结构保持与误差修正；
4. 深度—法线不一致度可能为低纹理区域提供有效残差信号；
5. 原始深度辅助监督和零残差初始化可能减少训练初期退化；
6. 独立阶段参数可能优于共享参数，但会增加模型容量。

## 19. 风险与局限

### 19.1 图像边缘不等于几何边缘

纹理、污渍、阴影和曝光变化可能产生强图像边缘。缓解方式是使用图像—深度联合条件，而不是只依赖 Sobel 响应。

### 19.2 早期法线噪声

Stage1 分辨率低，预测法线可能不稳定。可降低 Stage1 SGER 强度、使用置信度门控或先进行基线预训练。

### 19.3 残差过度修正

无约束残差可能破坏原始代价体给出的合理深度。需要 `tanh` 幅度限制、残差正则和零初始化。

### 19.4 级联误差放大

错误的 $\widetilde D_1$ 或 $\widetilde D_2$ 会改变后续采样中心。需要分阶段启用反馈、默认分离梯度并监控真值范围覆盖率。

### 19.5 阈值和域迁移

固定边缘、置信度和曲率阈值可能依赖数据集。应进行阈值敏感性分析，或后续研究可学习阈值和连续区域概率。

### 19.6 计算和显存成本

全阶段 SGER 比 Stage3-only 增加更多二维卷积和中间图。应比较轻量卷积、参数共享以及仅在训练阶段保存诊断输出的方案。

## 20. 推荐研究顺序

1. 实现 Stage3-only `SGERBlock`，验证残差细化是否有独立收益；
2. 增加几何/边缘门控并完成 hard、soft、dual 消融；
3. 将 SGER 扩展到 Stage2，验证跨阶段采样收益；
4. 最后加入 Stage1，重点检查早期噪声和级联误差；
5. 完成共享/独立参数、损失项和残差正则消融；
6. 在 DTU 指标、边界质量、低纹理弧面和效率之间综合选择最终配置。

该顺序可以先验证最小可行模块，再逐步增加跨阶段耦合，降低一次性修改整个级联网络带来的定位难度。

## 21. 总结

SGER-CasMVSNet 保留 CasMVSNet 的核心多视图匹配能力，并把当前仅用于训练约束的自几何与边缘信息进一步转化为推理阶段可执行的细化机制。每个阶段先由代价体产生原始深度、置信度和预测法线，再由 SGER 提取深度法线、法线不一致度、图像/深度边缘和曲率，通过 Region A/B 双区域门控控制轻量残差 CNN 的修正位置与幅度。

其关键创新假设不是“增加一个普通后处理 CNN”，而是：

> 用相机几何、网络自身预测、概率可靠性和边缘结构共同决定深度残差，并让细化结果参与后续级联采样。

该方案能否优于当前改进 CasMVSNet，需要通过严格的 Stage3-only/全阶段、规则/学习、硬/软门控、共享/独立参数和损失项消融验证。只有在深度、点云、边界质量及效率指标上获得一致改善，才能形成具有充分实验依据的 SGER-CasMVSNet。
