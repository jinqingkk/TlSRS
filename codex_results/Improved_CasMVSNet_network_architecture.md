# 改进的 CasMVSNet 网络架构详细说明

## 1. 网络总体定位

当前网络以 CasMVSNet 的三级粗到细多视图立体匹配框架为主体，在不改变深度假设采样、单应重投影、方差代价体和级联估计基本逻辑的前提下，增加了逐阶段表面法线预测分支，并在训练阶段引入置信度与边缘调制的几何约束。

网络的核心目标包括：

1. 保留 CasMVSNet 在大分辨率场景中的低显存、粗到细深度搜索优势；
2. 通过逐阶段 `NormalHead` 显式学习表面方向；
3. 利用预测深度、预测法线、相机内参、概率体置信度、图像边缘、深度边缘和曲率构造自几何监督；
4. 在可靠、非边缘或特定结构区域内施加不同强度的约束，改善低纹理区域深度漂移和大面积弧面不稳定问题；
5. 减少遮挡、拼接缝、孔洞和真实几何边缘附近的错误平滑。

当前实际生效的导入关系为：

```text
models/__init__.py
    └── models/cas_mvsnet.py
            └── from .module import *
```

因此，`models/module.py` 是当前主网络使用的共享模块；`models/module1.py` 是邻近的备用实现，**不在当前实际前向路径中**。

## 2. 网络总体结构

当前前向过程可以概括为：

```text
多视图图像 + 分阶段相机矩阵 + 初始深度范围
                    │
                    ▼
       每个视图独立通过 FeatureNet(FPN)
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
  Stage1 特征   Stage2 特征   Stage3 特征
   1/4, 32ch     1/2, 16ch      1×, 8ch
       │            ▲            ▲
       ▼            │            │
 Stage1 DepthNet ─D1┘            │
       │                          │
       ├── C1                     │
       └── NormalHead1 → N1       │
                                  │
             Stage2 DepthNet ─D2──┘
                    │
                    ├── C2
                    └── NormalHead2 → N2

                           Stage3 DepthNet
                                  │
                                  ├── D3（最终深度）
                                  ├── C3（最终置信度）
                                  └── NormalHead3 → N3（最终法线）
```

其中 $D_s$、$C_s$、$N_s$ 分别表示第 $s$ 阶段的预测深度、光度置信度和预测法线。

## 3. 输入与输出契约

### 3.1 输入

`CascadeMVSNet.forward(imgs, proj_matrices, depth_values)` 接收三个主要输入。

| 输入 | 典型形状 | 含义 |
|---|---|---|
| `imgs` | $(B,N,3,H,W)$ | 一个参考视图和 $N-1$ 个源视图组成的图像组 |
| `proj_matrices[stage_s]` | $(B,N,2,4,4)$ | 第 $s$ 阶段各视图的外参和缩放内参；索引 0 为外参，索引 1 为内参 |
| `depth_values` | $(B,D_0)$ | 数据集给出的初始全局深度假设范围 |

符号说明：

- $B$：批大小；
- $N$：视图数量；
- $H,W$：输入图像高度和宽度；
- $D_0$：数据加载阶段的初始深度采样数量。

视图 0 固定作为参考视图，其余视图作为源视图。

### 3.2 输出

输出 `outputs` 同时包含分阶段结果和最新阶段的顶层别名：

```python
outputs = {
    "stage1": {
        "depth": D1,
        "photometric_confidence": C1,
        "normal": N1,
    },
    "stage2": {
        "depth": D2,
        "photometric_confidence": C2,
        "normal": N2,
    },
    "stage3": {
        "depth": D3,
        "photometric_confidence": C3,
        "normal": N3,
    },
    "depth": D3,
    "photometric_confidence": C3,
    "normal": N3,
}
```

每处理完一个阶段，`outputs.update(outputs_stage)` 都会覆盖顶层键。因此默认三级网络的顶层 `depth`、`photometric_confidence` 和 `normal` 对应 Stage3 的全分辨率结果。

## 4. FeatureNet：多尺度 FPN 特征金字塔

### 4.1 自底向上的特征提取

每个视图独立通过同一个 `FeatureNet`，网络参数在所有视图之间共享。默认 `base_channels=8`，底层卷积结构为：

| 模块 | 主要操作 | 输出形状 |
|---|---|---|
| `conv0` | 两个 $3\times3$、步长 1 的二维卷积 | $(B,8,H,W)$ |
| `conv1` | 一个 $5\times5$、步长 2 卷积，随后两个 $3\times3$ 卷积 | $(B,16,H/2,W/2)$ |
| `conv2` | 一个 $5\times5$、步长 2 卷积，随后两个 $3\times3$ 卷积 | $(B,32,H/4,W/4)$ |

### 4.2 自顶向下的 FPN 融合

默认 `arch_mode="fpn"`。Stage1 直接由最深层 `conv2` 经 $1\times1$ 卷积得到：

$$
F_1=\operatorname{Conv}_{1\times1}(\operatorname{conv2}).
$$

Stage2 将 Stage1 内部特征上采样两倍，并与 `conv1` 经 $1\times1$ 投影后的特征相加：

$$
F_2=\operatorname{Conv}_{3\times3}
\left[
\operatorname{Up}_2(F_1^{\mathrm{inner}})
+\operatorname{Conv}_{1\times1}(\operatorname{conv1})
\right].
$$

Stage3 再次上采样，并与 `conv0` 的投影特征融合：

$$
F_3=\operatorname{Conv}_{3\times3}
\left[
\operatorname{Up}_2(F_2^{\mathrm{inner}})
+\operatorname{Conv}_{1\times1}(\operatorname{conv0})
\right].
$$

默认输出为：

| 阶段 | 分辨率 | 通道数 | 作用 |
|---|---:|---:|---|
| Stage1 | $H/4\times W/4$ | 32 | 大范围、低分辨率粗深度估计 |
| Stage2 | $H/2\times W/2$ | 16 | 缩小范围后的中尺度估计 |
| Stage3 | $H\times W$ | 8 | 全分辨率精细深度估计 |

FPN 同时保留深层语义信息和浅层空间细节，为级联估计提供逐步提高分辨率的特征。

## 5. 三级粗到细深度范围采样

### 5.1 全局深度范围

首先从输入 `depth_values` 记录全局最小和最大深度：

$$
d_{\min}=d_0,
\qquad
d_{\max}=d_{D_0-1}.
$$

代码定义基础深度间隔：

$$
\Delta d_{\mathrm{base}}=
\frac{d_{\max}-d_{\min}}{D_0}.
$$

### 5.2 Stage1：全局均匀采样

Stage1 没有上一阶段预测，因此 `cur_depth=depth_values`。`get_depth_range_samples()` 使用输入深度范围的两个端点，重新均匀采样 $D_1=48$ 个候选深度：

$$
d_{1,k}=d_{\min}
+k\frac{d_{\max}-d_{\min}}{D_1-1},
\qquad k=0,\ldots,D_1-1.
$$

该深度序列在空间维复制到所有像素，形成 $(B,D_1,H,W)$ 的深度假设张量，之后再三线性缩放到 Stage1 分辨率。

### 5.3 Stage2 和 Stage3：局部自适应采样

后续阶段以上一阶段深度作为每个像素的局部中心。对于第 $s$ 阶段：

$$
\delta_s=r_s\Delta d_{\mathrm{base}},
$$

$$
d_{s,\min}(\mathbf p)=
D_{s-1}(\mathbf p)-\frac{D_s^{\#}}{2}\delta_s,
$$

$$
d_{s,\max}(\mathbf p)=
D_{s-1}(\mathbf p)+\frac{D_s^{\#}}{2}\delta_s,
$$

$$
d_{s,k}(\mathbf p)=d_{s,\min}(\mathbf p)
+k\frac{d_{s,\max}(\mathbf p)-d_{s,\min}(\mathbf p)}{D_s^{\#}-1}.
$$

默认参数为：

| 阶段 | 深度假设数 $D_s^{\#}$ | 间隔倍率 $r_s$ | 空间尺度 |
|---|---:|---:|---:|
| Stage1 | 48 | 4 | $1/4$ |
| Stage2 | 32 | 2 | $1/2$ |
| Stage3 | 8 | 1 | $1$ |

深度范围随阶段逐渐收窄，空间分辨率逐渐提高，因此网络先解决大范围深度搜索，再专注于局部精细优化。

需要注意，虽然采样函数接收全局 `min_depth` 和 `max_depth` 参数，但当前 `get_cur_depth_range_samples()` 的有效代码没有对局部上下界执行 `clamp`；对应裁剪代码处于注释状态。因此，边界像素的局部候选深度理论上可能短暂超出输入全局范围。

### 5.4 梯度传播方式

默认 `grad_method="detach"`：

```python
cur_depth = depth.detach()
```

这意味着上一阶段深度只作为下一阶段采样中心，不通过采样范围向前一阶段传递梯度。若使用 `undetach`，采样中心保留计算图，但会增加跨阶段耦合和显存开销。

## 6. DepthNet：每阶段深度估计

每个阶段都调用同一个无参数 `DepthNet` 流程，但默认使用独立的 `CostRegNet`。

### 6.1 参考视图和源视图

阶段特征列表写为：

$$
\{F_{s,0},F_{s,1},\ldots,F_{s,N-1}\},
$$

其中 $F_{s,0}$ 是参考特征，其他为源特征。参考特征沿深度维复制：

$$
V_{s,0}(k,\mathbf p)=F_{s,0}(\mathbf p).
$$

### 6.2 投影矩阵组合

每个视图的阶段相机数据包含外参 $E_i$ 和内参 $K_{s,i}$。代码组合为

$$
Q_{s,i}=K_{s,i}E_i.
$$

相对投影变换为

$$
T_{i\leftarrow0}=Q_{s,i}Q_{s,0}^{-1}.
$$

将其前三行写为 $[R_{i0}\mid\mathbf t_{i0}]$。对参考像素齐次坐标 $\tilde{\mathbf p}=[u,v,1]^\mathsf T$ 和深度假设 $d_{s,k}$：

$$
\mathbf q_{i,k}=R_{i0}
\left(d_{s,k}\tilde{\mathbf p}\right)
+\mathbf t_{i0},
$$

$$
\mathbf p_{i,k}=
\left(
\frac{q_x}{q_z},
\frac{q_y}{q_z}
\right).
$$

### 6.3 可微单应重投影

将投影坐标归一化到 $[-1,1]$ 后，通过双线性 `grid_sample` 获得源视图重投影特征：

$$
\widetilde F_{s,i}(k,\mathbf p)=
F_{s,i}(\mathbf p_{i,k}).
$$

同一源特征会针对所有深度假设生成一个 $(B,C_s,D_s^{\#},H_s,W_s)$ 的特征体。

### 6.4 方差代价体

参考体和所有重投影源特征体通过方差聚合：

$$
V_s=
\frac{1}{N}\sum_{i=0}^{N-1}\widetilde F_{s,i}^{,2}
-
\left(
\frac{1}{N}\sum_{i=0}^{N-1}\widetilde F_{s,i}
\right)^2.
$$

若深度假设正确，不同视图特征会在参考坐标系中较好对齐，跨视图方差相对较小。方差聚合不依赖固定视图数，并对视图顺序不敏感。

## 7. CostRegNet：三维代价正则化

原始方差代价体包含局部噪声和匹配歧义，需要利用深度维和空间维上下文进行正则化。

`CostRegNet` 是一个三维编码器—解码器：

```text
输入代价体
  │
  ├─ Conv3D(base)
  ├─ stride-2 Conv3D → 2×base
  ├─ stride-2 Conv3D → 4×base
  ├─ stride-2 Conv3D → 8×base
  │
  ├─ Deconv3D + 4×base 编码特征
  ├─ Deconv3D + 2×base 编码特征
  ├─ Deconv3D + base 编码特征
  │
  └─ 3×3×3 Conv3D → 单通道深度得分体
```

编码路径逐步扩大三维感受野，解码路径通过三维反卷积恢复分辨率，并与对应尺度编码特征相加：

$$
X_4=C_4+\operatorname{Deconv}(X_8),
$$

$$
X_2=C_2+\operatorname{Deconv}(X_4),
$$

$$
X_1=C_1+\operatorname{Deconv}(X_2).
$$

最后输出形状为 $(B,1,D_s^{\#},H_s,W_s)$，压缩通道维后得到深度得分 $S_s$。

默认 `share_cr=False`，三个阶段分别使用输入通道为 32、16、8 的 `CostRegNet`，基础通道均为 8。

## 8. 概率体、深度回归与置信度

### 8.1 深度概率体

对得分体沿深度维做 softmax：

$$
P_s(k,\mathbf p)=
\frac{\exp S_s(k,\mathbf p)}
{\sum_j\exp S_s(j,\mathbf p)}.
$$

### 8.2 深度回归

预测深度为离散深度假设的概率期望：

$$
D_s(\mathbf p)=
\sum_kP_s(k,\mathbf p)d_{s,k}(\mathbf p).
$$

相比直接取最大概率索引，概率回归可以获得亚采样间隔的连续深度，并保持可微。

### 8.3 光度置信度

先估计概率索引期望并转为整数索引：

$$
\hat k_s(\mathbf p)=
\left\lfloor\sum_k kP_s(k,\mathbf p)\right\rfloor.
$$

在该索引附近累加四个深度概率：

$$
C_s(\mathbf p)=
\sum_{j=-1}^{2}P_s(\hat k_s+j,\mathbf p).
$$

概率越集中，$C_s$ 越高。置信度在 `torch.no_grad()` 中计算，主要用于推理筛选、几何掩码和损失权重调制。

## 9. 逐阶段 NormalHead 法线分支

### 9.1 分支输入

每个阶段都有独立的 `NormalHead`。输入由三个部分拼接：

$$
Z_s=
\operatorname{Concat}
\left(F_{s,0},D_s,C_s\right).
$$

其中深度和置信度先扩展通道维，并在必要时双线性缩放到参考特征分辨率。输入通道数为：

| 阶段 | 参考特征通道 | 深度 | 置信度 | 总输入通道 |
|---|---:|---:|---:|---:|
| Stage1 | 32 | 1 | 1 | 34 |
| Stage2 | 16 | 1 | 1 | 18 |
| Stage3 | 8 | 1 | 1 | 10 |

### 9.2 分支结构

每个 `NormalHead` 包含：

1. $3\times3$ 卷积：输入通道 $\rightarrow32$；
2. $3\times3$ 卷积：$32\rightarrow32$；
3. $3\times3$ 输出卷积：$32\rightarrow3$；
4. 沿三通道执行 $L_2$ 归一化。

$$
\widehat N_s=f_{\theta_s}(Z_s),
\qquad
N_s^{\mathrm{pred}}=
\frac{\widehat N_s}
{\|\widehat N_s\|_2+\varepsilon}.
$$

输出形状为 $(B,3,H_s,W_s)$，每个像素是单位表面法线。

### 9.3 训练信号

当前数据流程没有使用真值法线。`NormalHead` 的主要显式训练信号来自预测法线与深度反算法线之间的一致性损失。这样可以在不增加法线标注的情况下学习表面方向。

## 10. Stage1 → Stage2 → Stage3 完整数据流

| 项目 | Stage1 | Stage2 | Stage3 |
|---|---|---|---|
| 特征分辨率 | $1/4$ | $1/2$ | $1$ |
| 特征通道 | 32 | 16 | 8 |
| 深度假设数 | 48 | 32 | 8 |
| 深度间隔倍率 | 4 | 2 | 1 |
| 采样中心 | 全局深度范围 | 上采样后的 $D_1$ | 上采样后的 $D_2$ |
| 代价正则化 | 独立 `CostRegNet[0]` | 独立 `CostRegNet[1]` | 独立 `CostRegNet[2]` |
| 法线头 | `NormalHead[0]` | `NormalHead[1]` | `NormalHead[2]` |
| 输出 | $D_1,C_1,N_1$ | $D_2,C_2,N_2$ | $D_3,C_3,N_3$ |

Stage1 负责确定全局深度位置；Stage2 在局部范围内提高空间分辨率；Stage3 使用最少的深度假设完成全分辨率细化。该设计将高成本的大范围搜索放在低分辨率阶段，将高分辨率计算限制在窄深度区间内。

## 11. 几何信息来源

训练阶段使用以下几何信息：

| 几何信息 | 来源 | 用途 |
|---|---|---|
| 预测深度 $D_s$ | 深度概率期望 | 法线、深度边缘、曲率和全部深度正则 |
| 深度反算法线 $N_s^D$ | $D_s$、相机内参、三维点切向量 | 深度—法线一致性 |
| 网络预测法线 $N_s^{pred}$ | `NormalHead(F_ref,D,C)` | 学习显式表面方向 |
| 多视图几何 | 相机矩阵和单应重投影 | 构建方差代价体 |
| 光度置信度 $C_s$ | 深度概率体局部四深度窗 | 硬掩码和软权重 |
| 图像边缘 | 一阶差分或 Sobel | 保护外观/结构边界 |
| 深度边缘 | 均值归一化深度的一阶差分 | 识别深度不连续 |
| 曲率 | 归一化深度的二阶差分 | 识别弯曲区域并约束二阶连续性 |

需要强调：多视图重投影用于特征对齐和代价体构建，**不是独立的重投影损失**。

## 12. 几何掩码、连续权重与双区域机制

### 12.1 高置信非边缘硬掩码

深度—法线一致性只在有效、高置信且非图像边缘像素计算：

$$
M_s^{\mathrm{smooth}}=
M_s\land[C_s>\tau_c]
\land[E_s<\tau_e].
$$

默认 $\tau_c=0.8$、$\tau_e=0.05$。

### 12.2 连续几何权重

置信度和边缘分别形成 sigmoid 权重：

$$
w_s^{\mathrm{conf}}=
\sigma\left(k_c(C_s-c_0)\right),
$$

$$
w_s^{\mathrm{edge}}=
\sigma\left(k_e(e_0-E_s)\right).
$$

最终连续几何权重为：

$$
w_s^{\mathrm{geo}}=M_s
\left[w_{\min}+(1-w_{\min})
w_s^{\mathrm{conf}}w_s^{\mathrm{edge}}\right].
$$

默认 $c_0=0.65$、$k_c=10$、$e_0=0.25$、$k_e=10$、$w_{\min}=0.05$。

### 12.3 Region A/B 双区域

代码通过 Sobel 图像边缘 $E_s^{S}$、归一化深度梯度 $G_s^D$ 和曲率 $\kappa_s$ 构造区域：

$$
R_s^A=M_s\land
\left[
(E_s^{S}>\tau_I\land G_s^D>\tau_D)
\lor(\kappa_s>\tau_\kappa)
\right],
$$

$$
R_s^B=M_s\land\neg R_s^A.
$$

Region A 使用硬掩码；Region B 使用置信度和图像边缘衰减形成软权重：

$$
w_s^B=\mathbf1_{R_s^B}
\sigma\left(k_c(C_s-c_0)\right)
\exp(-k_{\mathrm{smooth}}E_s^{S}).
$$

区域构造中的深度和置信度会被 `detach()`，阈值划分本身不参与反向传播。

## 13. 训练损失

### 13.1 多阶段深度监督

每个阶段在有效像素上使用 Smooth L1：

$$
\mathcal L_{\mathrm{depth}}^s=
\operatorname{SmoothL1}(D_s-D_s^{gt}).
$$

默认阶段权重：

$$
(w_1^D,w_2^D,w_3^D)=(0.5,1.0,2.0).
$$

### 13.2 深度—法线一致性

由深度和阶段内参恢复三维点：

$$
X_s(u,v)=D_s(u,v)K_s^{-1}[u,v,1]^\mathsf T.
$$

局部切向量叉乘得到深度反算法线：

$$
N_s^D=\operatorname{normalize}
\left[
(X_{u+1,v}-X_{u,v})
\times
(X_{u,v+1}-X_{u,v})
\right].
$$

一致性损失为：

$$
\mathcal L_{\mathrm{dn}}^s=
\operatorname{Mean}_{M_s^{\mathrm{smooth}}}
\left[
1-\left|N_s^{pred}\cdot N_s^D\right|
\right].
$$

绝对值消除法线正负方向的二义性。

### 13.3 法线平滑损失

从深度梯度构造简化法线：

$$
\widetilde N_s=
\operatorname{normalize}[-\partial_xD_s,-\partial_yD_s,1].
$$

相邻有效像素的法线余弦差形成平滑项：

$$
\mathcal L_{\mathrm{ns}}^s=
\operatorname{Mean}
\left(1-\widetilde N_s(\mathbf p)
\cdot\widetilde N_s(\mathbf p+\mathbf e_x)\right)
+
\operatorname{Mean}
\left(1-\widetilde N_s(\mathbf p)
\cdot\widetilde N_s(\mathbf p+\mathbf e_y)\right).
$$

### 13.4 曲率连续性

先使用有效区域均值归一化深度：

$$
D_s^n=\frac{D_s}{\operatorname{Mean}_{M_s}(D_s)+\varepsilon}.
$$

离散二阶差分：

$$
\Delta_{xx}D_s^n=D_s^n(u+1,v)-2D_s^n(u,v)+D_s^n(u-1,v),
$$

$$
\Delta_{yy}D_s^n=D_s^n(u,v+1)-2D_s^n(u,v)+D_s^n(u,v-1).
$$

默认启用双区域曲率：

$$
\mathcal L_{\mathrm{curv}}^s=
\lambda_A\mathcal L_A^s+
\lambda_B\mathcal L_B^s,
$$

其中 $\lambda_A=1.5$、$\lambda_B=1.0$。Region A 使用硬三点模板，Region B 的三点模板权重取相邻三点软权重的最小值。

### 13.5 边缘感知平滑

图像梯度控制深度一阶平滑强度：

$$
\mathcal L_{\mathrm{edge}}^s=
\operatorname{Mean}_{M_{s,x}}
\left(|\partial_xD_s^n|e^{-|\partial_xI|}\right)
+
\operatorname{Mean}_{M_{s,y}}
\left(|\partial_yD_s^n|e^{-|\partial_yI|}\right).
$$

在纹理平坦区域强平滑，在图像边缘附近降低平滑强度。

## 14. 多阶段总损失

以一基阶段编号 $s\in\{1,2,3\}$ 表示，当前代码对应的总目标为：

$$
\begin{aligned}
\mathcal L_{\mathrm{total}}
=\sum_{s=1}^{3}\Bigg[&
w_s^D\mathcal L_{\mathrm{depth}}^s
+\frac{\lambda_{\mathrm{ns}}}{2^{s-1}}
\mathcal L_{\mathrm{ns}}^s\\
&+\frac{\lambda_{\mathrm{edge}}}{2^{s-1}}
\mathcal L_{\mathrm{edge}}^s
+\frac{\lambda_{\mathrm{dn}}}{2^{s-1}}
\mathcal L_{\mathrm{dn}}^s\\
&+\lambda_{\mathrm{curv}}\frac{s}{2}
\mathcal L_{\mathrm{curv}}^s
\Bigg].
\end{aligned}
$$

默认基础权重为：

| 损失 | 参数 | 默认值 |
|---|---|---:|
| 法线平滑 | `normal_smooth_loss_weight` | 0.02 |
| 曲率连续 | `curv_loss_weight` | 0.005 |
| 边缘感知平滑 | `edge_smooth_loss_weight` | 0.005 |
| 深度—法线一致性 | `depth_normal_loss_weight` | 0.03 |

需要注意两种不同的阶段权重规则：

- 法线平滑、边缘感知平滑和深度—法线一致性依次乘以 $1、1/2、1/4$；
- 当前曲率项依次乘以 $0.5、1.0、1.5$，而不是相同的二分之一衰减。

## 15. 梯度传播与 detach 行为

| 位置 | 默认行为 | 影响 |
|---|---|---|
| 上一阶段深度 → 下一阶段采样中心 | `detach()` | 下一阶段采样不反向影响上一阶段深度 |
| 光度置信度 | 在 `torch.no_grad()` 中生成 | 作为可靠性依据，不通过该分支反传 |
| 连续几何权重中的置信度 | `detach()` | 权重选择与置信度梯度解耦 |
| Region A/B 构造中的深度 | `detach()` | 阈值区域划分不通过深度边缘反传 |
| 深度均值归一化 | 均值 `detach()` | 避免全局尺度统计参与梯度 |
| 曲率/边缘/法线损失中的预测深度 | 不分离 | 几何损失仍能更新深度网络 |
| 深度—法线一致性中的预测法线 | 不分离 | 更新 `NormalHead` |

这种设计将离散区域选择和可靠性估计视为固定调制信号，同时保持实际几何残差对网络参数可导。

## 16. 训练与推理

### 16.1 训练

`train.py` 默认构造：

```python
CascadeMVSNet(
    refine=False,
    ndepths=[48, 32, 8],
    depth_interals_ratio=[4, 2, 1],
    share_cr=False,
    cr_base_chs=[8, 8, 8],
    grad_method="detach",
)
```

训练计算所有启用的深度和几何损失，并记录 `normal_depth_cos`、`smooth_mask_ratio`、`geometry_weight_mean`、`region_A_ratio`、`region_B_ratio`、`curv_loss_A` 和 `curv_loss_B` 等指标。

### 16.2 推理

`test.py` 同样使用 `refine=False`。推理阶段执行：

1. 多视图 FPN 特征提取；
2. 三级深度估计；
3. 每阶段法线预测；
4. 输出 Stage3 深度、置信度和法线；
5. 后续保存深度/置信度，并按测试脚本完成过滤和融合。

几何损失只在训练时参与目标函数，不增加推理时的损失计算成本；`NormalHead` 会增加少量二维卷积计算。

## 17. 相对原始 CasMVSNet 的主要改进

### 17.1 逐阶段显式法线预测

原始主干主要输出深度和置信度；当前网络在三个级联阶段都增加 `NormalHead`，使每个尺度都有显式表面方向表示。

### 17.2 深度和法线的自监督闭环

预测深度通过相机内参反算法线，网络同时预测法线，两者通过余弦一致性互相约束。该机制不依赖真值法线。

### 17.3 置信度与边缘共同控制几何监督

网络不会在所有像素上等强度地施加几何约束。高置信非边缘硬掩码适用于法线一致性；连续权重和双区域机制用于曲率调制。

### 17.4 从一阶平滑扩展到二阶几何连续

边缘感知一阶平滑降低局部噪声，曲率二阶约束进一步稳定大面积连续曲面和弧面。

### 17.5 面向的主要问题

- 无纹理或弱纹理区域匹配不唯一；
- 大面积混凝土表面和弧面深度抖动；
- 局部概率分布不稳定导致深度漂移；
- 统一平滑导致拼接缝、孔洞和遮挡边界被模糊；
- 只依赖深度监督时局部表面方向缺乏显式约束。

## 18. 实现边界

1. `models/module.py` 是当前生效实现，`models/module1.py` 不参与当前模型。
2. `RefineNet` 类虽然存在，但训练和推理均设置 `refine=False`；该分支不属于当前默认网络输出。
3. 单独架构图中提出的独立 SEGER 残差细化模块是概念性扩展，**不在当前实际前向路径中**。
4. 多视图重投影用于构建代价体，当前没有独立的光度或几何重投影损失。
5. `NormalHead` 没有真值法线监督，主要依赖深度—法线一致性学习。
6. 最终深度仍是 Stage3 深度，而不是额外残差细化后的深度。

## 19. 类与函数对应关系

| 功能 | 类或函数 | 文件 |
|---|---|---|
| 级联网络组织 | `CascadeMVSNet` | `models/cas_mvsnet.py` |
| 单阶段深度流程 | `DepthNet` | `models/cas_mvsnet.py` |
| FPN 特征提取 | `FeatureNet` | `models/module.py` |
| 多视图单应重投影 | `homo_warping` | `models/module.py` |
| 三维代价正则化 | `CostRegNet` | `models/module.py` |
| 深度概率期望 | `depth_regression` | `models/module.py` |
| 逐阶段法线预测 | `NormalHead` | `models/module.py` |
| 相机感知深度法线 | `compute_normal_from_depth` | `models/module.py` |
| 图像边缘 | `image_gradient_magnitude`、`sobel_gradient_magnitude` | `models/module.py` |
| 深度边缘与曲率 | `depth_gradient_magnitude`、`curvature_magnitude` | `models/module.py` |
| 高置信非边缘掩码 | `build_smooth_mask` | `models/module.py` |
| 连续几何权重 | `build_geometry_weight` | `models/module.py` |
| Region A/B | `build_dual_region_geometry` | `models/module.py` |
| 深度—法线一致性 | `depth_normal_consistency_loss` | `models/module.py` |
| 法线平滑 | `normal_smooth_loss` | `models/module.py` |
| 曲率约束 | `curvature_loss`、`soft_curvature_loss`、`dual_region_curvature_loss` | `models/module.py` |
| 边缘感知平滑 | `edge_aware_smooth_loss` | `models/module.py` |
| 多阶段总损失 | `cas_mvsnet_loss` | `models/module.py` |
| 训练入口与默认参数 | 模型构造、`loss_kwargs` | `train.py` |
| 推理入口 | `save_scene_depth` | `test.py` |

## 20. 总结

当前改进网络保持了 CasMVSNet 的三级粗到细主体：FPN 提取多尺度特征，深度范围由全局逐步收窄到局部，多视图特征经单应重投影形成方差代价体，三维网络正则化后通过概率回归得到深度。

在此基础上，每个阶段增加法线预测分支，并在训练阶段将深度、法线、置信度、图像边缘、深度边缘和曲率组合为多层次几何约束。硬掩码控制约束区域，连续权重调节约束强度，Region A/B 机制进一步区分结构区域和连续表面。最终网络在不重写 CasMVSNet 核心代价体和级联框架的前提下，增强了对低纹理、大面积弧面及边界区域的几何建模能力。
