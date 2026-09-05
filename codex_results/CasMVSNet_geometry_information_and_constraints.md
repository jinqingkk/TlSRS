# 改进 CasMVSNet 中的几何信息来源、计算公式与约束机制

## 1. 文档目的

本文档说明当前改进版 CasMVSNet 中几何信息的主要来源、计算方式及其对应的训练约束。内容严格以当前生效的 `models/cas_mvsnet.py`、`models/module.py` 和 `train.py` 为依据，并采用适合论文方法章节的数学表达。

需要特别说明：**多视图重投影关系在当前实现中用于构建代价体和深度概率分布，但不是独立的重投影损失**。因此，多视图重投影属于几何证据来源，而当前显式损失主要包括深度监督、深度—法线一致性、法线平滑、曲率连续性和边缘感知平滑。

## 2. 符号约定

| 符号 | 含义 |
|---|---|
| $s\in\{1,2,3\}$ | 级联阶段编号，由粗到细 |
| $\mathbf p=(u,v)$ | 参考视图中的像素坐标 |
| $d_{s,k}(\mathbf p)$ | 第 $s$ 阶段、第 $k$ 个深度假设 |
| $S_s(k,\mathbf p)$ | 3D 代价正则化网络输出的深度得分 |
| $P_s(k,\mathbf p)$ | 对深度维做 softmax 后的概率体 |
| $D_s(\mathbf p)$ | 第 $s$ 阶段的预测深度 |
| $C_s(\mathbf p)$ | 由概率体计算的光度置信度 |
| $K_s$ | 缩放到第 $s$ 阶段分辨率的相机内参 |
| $Q_i=K_i[R_i\mid\mathbf t_i]$ | 第 $i$ 个视图的组合投影矩阵 |
| $F_i$ | 第 $i$ 个视图的特征图，$i=0$ 为参考视图 |
| $N_s^{\mathrm{pred}}$ | `NormalHead` 预测的单位法线 |
| $N_s^D$ | 由预测深度和相机内参反算的单位法线 |
| $M_s$ | 有效深度掩码 |
| $E_s$ | 参考图像边缘强度 |
| $G_s^D$ | 归一化深度梯度强度 |
| $\kappa_s$ | 归一化深度的离散曲率强度 |
| $\langle x\rangle_M$ | 在掩码或权重 $M$ 下的加权平均 |

对任意量 $x$，本文使用以下掩码平均和加权平均：

$$
\operatorname{Mean}_M(x)=
\frac{\sum_{\mathbf p}M(\mathbf p)x(\mathbf p)}
{\sum_{\mathbf p}M(\mathbf p)+\varepsilon},
$$

其中 $\varepsilon=10^{-6}$ 用于数值稳定。

## 3. 几何信息的来源与计算公式

### 3.1 深度概率体与预测深度

#### 信息来源

每个阶段首先根据当前深度范围构造离散深度假设，经多视图特征重投影、方差聚合和 3D 代价正则化后得到深度得分 $S_s$。沿深度维归一化：

$$
P_s(k,\mathbf p)=
\frac{\exp S_s(k,\mathbf p)}
{\sum_{j=1}^{D_s^{\#}}\exp S_s(j,\mathbf p)},
$$

其中 $D_s^{\#}$ 表示该阶段的深度假设数量，默认依次为 $48、32、8$。

深度回归采用概率期望：

$$
D_s(\mathbf p)=
\sum_{k=1}^{D_s^{\#}}P_s(k,\mathbf p)d_{s,k}(\mathbf p).
$$

#### 说明

概率期望比直接取最大概率深度更平滑，并允许梯度通过概率体传播至代价正则化网络和特征提取网络。后续法线、深度边缘和曲率信息都以 $D_s$ 为基础。

对应实现：`DepthNet.forward()`、`depth_regression()`。

### 3.2 多视图重投影关系

#### 信息来源

参考视图像素的齐次坐标记为

$$
\tilde{\mathbf p}=[u,v,1]^\mathsf T.
$$

在深度假设 $d$ 下，代码使用组合投影矩阵的相对变换

$$
T_{i\leftarrow0}=Q_iQ_0^{-1}
=
\begin{bmatrix}
R_{i0}&\mathbf t_{i0}\\
\mathbf 0^\mathsf T&1
\end{bmatrix}.
$$

源视图中的投影点按下式计算：

$$
\mathbf q_i(d,\mathbf p)
=R_{i0}\bigl(d\tilde{\mathbf p}\bigr)+\mathbf t_{i0},
$$

$$
\pi(\mathbf q_i)=
\left(
\frac{q_{i,x}}{q_{i,z}},
\frac{q_{i,y}}{q_{i,z}}
\right).
$$

随后通过双线性采样获得重投影特征：

$$
\widetilde F_i(d,\mathbf p)
=F_i\bigl(\pi(\mathbf q_i(d,\mathbf p))\bigr).
$$

#### 方差代价体

将参考特征和所有重投影源特征记为 $\widetilde F_i$，共 $N$ 个视图，则方差聚合为

$$
V_s(d,\mathbf p)=
\frac{1}{N}\sum_{i=0}^{N-1}\widetilde F_i(d,\mathbf p)^2
-
\left[
\frac{1}{N}\sum_{i=0}^{N-1}\widetilde F_i(d,\mathbf p)
\right]^2.
$$

#### 说明

正确深度假设应使不同视图的对应特征在参考坐标系内对齐，从而产生较小的跨视图方差。该关系是网络最主要的多视图几何证据。当前代码没有把像素重投影误差单独加入 `cas_mvsnet_loss`，因此它**不是独立的重投影损失**。

对应实现：`homo_warping()`、`DepthNet.forward()` 中的 `volume_variance`。

### 3.3 概率体置信度

#### 信息来源

先由概率分布估计连续深度索引：

$$
\bar k_s(\mathbf p)=
\sum_k kP_s(k,\mathbf p),
\qquad
\hat k_s(\mathbf p)=\left\lfloor\bar k_s(\mathbf p)\right\rfloor.
$$

代码在 $\hat k_s$ 附近对连续四个深度概率求和：

$$
C_s(\mathbf p)=
\sum_{j=-1}^{2}
P_s\bigl(\hat k_s(\mathbf p)+j,\mathbf p\bigr),
$$

越界位置按零处理，并将索引限制在有效深度范围内。

#### 说明

$C_s$ 衡量回归深度附近的概率集中程度。概率越集中，表示网络对当前深度估计越确定。当前实现通过 `torch.no_grad()` 计算置信度，因此置信度用于掩码和权重调制，不直接接收梯度。

对应实现：`DepthNet.forward()` 中的四深度窗平均池化与 `photometric_confidence`。

### 3.4 预测深度反算法线

#### 相机坐标点恢复

对于内参

$$
K_s=
\begin{bmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix},
$$

像素 $\mathbf p=(u,v)$ 对应的三维点为

$$
X_s(u,v)=D_s(u,v)K_s^{-1}[u,v,1]^\mathsf T,
$$

即

$$
X_s(u,v)=
\begin{bmatrix}
(u-c_x)D_s/f_x\\
(v-c_y)D_s/f_y\\
D_s
\end{bmatrix}.
$$

#### 切向量与法线

使用前向差分构造两个局部切向量：

$$
T_x(u,v)=X_s(u+1,v)-X_s(u,v),
$$

$$
T_y(u,v)=X_s(u,v+1)-X_s(u,v).
$$

深度反算法线为

$$
N_s^D(u,v)=
\frac{T_x(u,v)\times T_y(u,v)}
{\|T_x(u,v)\times T_y(u,v)\|_2+\varepsilon}.
$$

边界处复制最后一个有效差分以保持输出尺寸不变。

#### 说明

$N_s^D$ 是由深度和相机模型确定的几何法线，体现了预测深度在三维空间中的局部表面方向。它被用于深度—法线一致性约束。

对应实现：`compute_normal_from_depth()`。

### 3.5 网络预测法线

每个级联阶段均有一个独立 `NormalHead`。其输入为参考视图特征、当前阶段深度和光度置信度：

$$
Z_s(\mathbf p)=
\left[
F_{s,0}(\mathbf p),
D_s(\mathbf p),
C_s(\mathbf p)
\right].
$$

经过两层带归一化和激活的 $3\times3$ 卷积及一层三通道输出卷积：

$$
\widehat N_s=f_{\theta_s}(Z_s),
\qquad
N_s^{\mathrm{pred}}=
\frac{\widehat N_s}
{\|\widehat N_s\|_2+\varepsilon}.
$$

#### 说明

预测法线从图像特征、深度形状和深度可靠性中联合学习表面方向。当前实现没有使用真值法线监督；`NormalHead` 主要通过深度—法线一致性损失获得训练信号。

对应实现：`NormalHead.forward()`、`CascadeMVSNet.forward()`。

### 3.6 图像边缘

当前代码包含两类图像边缘计算。

#### 一阶差分边缘

对参考图像三个通道取前向差分并求通道平均：

$$
g_x^I(u,v)=
\frac{1}{3}\sum_{c=1}^3
\left|I_c(u+1,v)-I_c(u,v)\right|,
$$

$$
g_y^I(u,v)=
\frac{1}{3}\sum_{c=1}^3
\left|I_c(u,v+1)-I_c(u,v)\right|.
$$

实现中将相邻差分写回差分两侧，并通过逐位置最大值得到边缘强度 $E_s^{\mathrm{diff}}$。该边缘用于高置信非边缘掩码和连续几何权重。

#### Sobel 边缘

先将图像转成通道均值灰度图 $I_g$，再使用

$$
S_x=
\begin{bmatrix}
-1&0&1\\-2&0&2\\-1&0&1
\end{bmatrix},
\qquad
S_y=
\begin{bmatrix}
-1&-2&-1\\0&0&0\\1&2&1
\end{bmatrix}.
$$

Sobel 边缘幅值为

$$
E_s^{\mathrm{Sobel}}=
\sqrt{(S_x*I_g)^2+(S_y*I_g)^2+10^{-12}}.
$$

该边缘用于 Region A/B 双区域划分。

对应实现：`image_gradient_magnitude()`、`sobel_gradient_magnitude()`。

### 3.7 深度边缘

为减小不同场景绝对深度尺度的影响，先用有效区域均值归一化：

$$
\bar D_s=
\operatorname{Mean}_{M_s}(D_s),
\qquad
D_s^n=\frac{D_s}{\bar D_s+\varepsilon}.
$$

其中 $\bar D_s$ 在代码中被 `detach()`，不参与反向传播。深度边缘由归一化深度的一阶差分得到：

$$
G_{s,x}^D(u,v)=
\left|D_s^n(u+1,v)-D_s^n(u,v)\right|,
$$

$$
G_{s,y}^D(u,v)=
\left|D_s^n(u,v+1)-D_s^n(u,v)\right|.
$$

代码将差分值分配给相邻像素并取最大值，得到 $G_s^D$。较大的 $G_s^D$ 通常对应物体边界、遮挡边界或深度不连续区域。

对应实现：`depth_gradient_magnitude()`。

### 3.8 深度曲率

使用归一化深度的二阶中心差分近似局部曲率：

$$
\kappa_{s,x}(u,v)=
\left|
D_s^n(u+1,v)-2D_s^n(u,v)+D_s^n(u-1,v)
\right|,
$$

$$
\kappa_{s,y}(u,v)=
\left|
D_s^n(u,v+1)-2D_s^n(u,v)+D_s^n(u,v-1)
\right|.
$$

曲率强度图取两个方向的局部最大响应：

$$
\kappa_s(u,v)=
\max\bigl(\kappa_{s,x}(u,v),\kappa_{s,y}(u,v)\bigr).
$$

#### 说明

一阶梯度描述深度变化速度，二阶差分描述变化速度的变化。曲率可用于识别高弯曲结构，也可用于约束连续曲面的二阶平滑性。

对应实现：`curvature_magnitude()`、`curvature_loss()`。

## 4. 几何掩码与区域调制

### 4.1 高置信非边缘硬掩码

深度—法线一致性只在有效、高置信且非图像边缘区域计算：

$$
M_s^{\mathrm{smooth}}=
M_s
\land [C_s>\tau_c]
\land [E_s^{\mathrm{diff}}<\tau_e].
$$

默认阈值为 $\tau_c=0.8$、$\tau_e=0.05$。

该掩码的作用是避免低置信区域、遮挡区域和真实几何边缘对法线一致性产生错误监督。

对应实现：`build_smooth_mask()`。

### 4.2 连续几何权重

与硬筛选相比，连续权重保留所有有效像素，但根据置信度和图像边缘平滑调节强度：

$$
w_s^{\mathrm{conf}}(\mathbf p)=
\sigma\bigl(k_c[C_s(\mathbf p)-c_0]\bigr),
$$

$$
w_s^{\mathrm{edge}}(\mathbf p)=
\sigma\bigl(k_e[e_0-E_s^{\mathrm{diff}}(\mathbf p)]\bigr),
$$

$$
w_s^{\mathrm{geo}}(\mathbf p)=
M_s(\mathbf p)
\left[
w_{\min}+(1-w_{\min})
w_s^{\mathrm{conf}}(\mathbf p)
w_s^{\mathrm{edge}}(\mathbf p)
\right].
$$

当前默认值为

$$
c_0=0.65,\quad k_c=10,\quad
e_0=0.25,\quad k_e=10,\quad
w_{\min}=0.05.
$$

置信度在该函数中显式 `detach()`；图像边缘来自输入图像。因此，权重本身不通过置信度分支接收梯度，而被加权的深度损失仍可向深度网络反向传播。

对应实现：`build_geometry_weight()`。

### 4.3 Region A/B 双区域调制

首先定义三类二值条件：

$$
M_s^I=[E_s^{\mathrm{Sobel}}>\tau_I],
$$

$$
M_s^D=[G_s^D>\tau_D],
$$

$$
M_s^\kappa=[\kappa_s>\tau_\kappa].
$$

Region A 定义为有效区域内的图像—深度联合边缘或高曲率区域：

$$
R_s^A=M_s\land
\left[(M_s^I\land M_s^D)\lor M_s^\kappa\right].
$$

Region B 是其在有效区域内的补集：

$$
R_s^B=M_s\land\neg R_s^A.
$$

Region B 使用连续软权重：

$$
w_s^B=\mathbf 1_{R_s^B}
\,\sigma\bigl(k_c[C_s-c_0]\bigr)
\exp(-k_{\mathrm{smooth}}E_s^{\mathrm{Sobel}}).
$$

默认参数为

$$
\tau_I=0.25,\quad
\tau_D=0.02,\quad
\tau_\kappa=0.02,\quad
k_{\mathrm{smooth}}=2.
$$

构造区域时，预测深度和置信度均被 `detach()`。这意味着区域选择不会通过阈值操作向深度或置信度分支反向传播，但区域内计算的曲率损失仍对原始预测深度求梯度。

对应实现：`build_dual_region_geometry()`。

## 5. 几何约束及计算公式

### 5.1 深度—法线一致性约束

将网络预测法线和深度反算法线都归一化后，使用方向无关的余弦一致性：

$$
\rho_s(\mathbf p)=
\left|
N_s^{\mathrm{pred}}(\mathbf p)\cdot
N_s^D(\mathbf p)
\right|.
$$

绝对值使法线正负方向翻转不影响局部表面方向的一致性。损失为

$$
\mathcal L_{\mathrm{dn}}^s=
\operatorname{Mean}_{M_s^{\mathrm{smooth}}}
\left(1-\rho_s\right).
$$

#### 作用

- 使 `NormalHead` 输出与深度几何相容；
- 利用法线方向约束大面积低纹理弧面的局部形状；
- 通过高置信非边缘掩码减少遮挡和真实边界处的错误约束。

对应实现：`depth_normal_consistency_loss()`。

### 5.2 法线平滑约束

该项不是直接使用 `NormalHead` 输出，而是从深度梯度构造简化法线：

$$
\widetilde N_s(u,v)=
\operatorname{normalize}
\left[-\partial_xD_s,-\partial_yD_s,1\right]^\mathsf T.
$$

在相邻有效像素间约束法线方向一致：

$$
\mathcal L_{\mathrm{ns}}^s=
\operatorname{Mean}_{M_{s,x}}
\left[1-\widetilde N_s(u,v)\cdot
\widetilde N_s(u+1,v)\right]
+
\operatorname{Mean}_{M_{s,y}}
\left[1-\widetilde N_s(u,v)\cdot
\widetilde N_s(u,v+1)\right],
$$

其中 $M_{s,x}$ 和 $M_{s,y}$ 要求差分两端均为有效像素。

#### 作用

该项抑制局部表面方向的高频抖动，适合改善连续混凝土表面或弧面上的深度噪声。它与相机内参感知的 $N_s^D$ 不同，是一个轻量的深度局部平滑正则。

对应实现：`get_depth_normals()`、`normal_smooth_loss()`。

### 5.3 曲率连续性约束

#### 基础曲率损失

对归一化深度二阶差分取绝对值：

$$
\mathcal L_{\mathrm{curv,raw}}^s=
\operatorname{Mean}_{M_{s,xxx}}
|\Delta_{xx}D_s^n|
+
\operatorname{Mean}_{M_{s,yyy}}
|\Delta_{yy}D_s^n|,
$$

其中三点掩码要求二阶差分模板内的三个像素均有效。

#### 软权重曲率损失

若像素权重为 $w_s^{\mathrm{geo}}$，则一个三点模板的权重取三点最小值：

$$
w_{s,x}^{(3)}(u,v)=
\min\{w_s(u-1,v),w_s(u,v),w_s(u+1,v)\},
$$

$$
w_{s,y}^{(3)}(u,v)=
\min\{w_s(u,v-1),w_s(u,v),w_s(u,v+1)\}.
$$

于是

$$
\mathcal L_{\mathrm{curv,soft}}^s=
\operatorname{Mean}_{w_{s,x}^{(3)}}|\Delta_{xx}D_s^n|
+
\operatorname{Mean}_{w_{s,y}^{(3)}}|\Delta_{yy}D_s^n|.
$$

取最小值相当于硬掩码三点逻辑与的连续推广：只要模板中有一点不可靠，该模板的曲率权重就会相应降低。

#### 双区域曲率损失

Region A 使用硬掩码，Region B 使用软权重：

$$
\mathcal L_A^s=
\operatorname{Mean}_{R_{s,x}^{A,(3)}}|\Delta_{xx}D_s^n|
+
\operatorname{Mean}_{R_{s,y}^{A,(3)}}|\Delta_{yy}D_s^n|,
$$

$$
\mathcal L_B^s=
\operatorname{Mean}_{w_{s,x}^{B,(3)}}|\Delta_{xx}D_s^n|
+
\operatorname{Mean}_{w_{s,y}^{B,(3)}}|\Delta_{yy}D_s^n|,
$$

$$
\mathcal L_{\mathrm{curv,dual}}^s=
\lambda_A\mathcal L_A^s+
\lambda_B\mathcal L_B^s.
$$

默认 $\lambda_A=1.5$、$\lambda_B=1.0$。

#### 作用

曲率损失约束深度表面的二阶连续性。双区域机制使结构显著区域和连续表面采用不同的约束方式，避免用单一全局权重处理所有区域。

对应实现：`curvature_loss()`、`soft_curvature_loss()`、`dual_region_curvature_loss()`。

### 5.4 边缘感知平滑约束

先将参考图像缩放到当前深度分辨率，并对归一化深度计算一阶差分。图像梯度产生指数衰减权重：

$$
w_x^I=\exp(-g_x^I),
\qquad
w_y^I=\exp(-g_y^I).
$$

边缘感知平滑损失为

$$
\mathcal L_{\mathrm{edge}}^s=
\operatorname{Mean}_{M_{s,x}}
\left(|\partial_xD_s^n|e^{-g_x^I}\right)
+
\operatorname{Mean}_{M_{s,y}}
\left(|\partial_yD_s^n|e^{-g_y^I}\right).
$$

#### 作用

在图像平坦区域，$g^I$ 较小，权重接近 1，因而较强地抑制深度噪声；在图像边缘处，权重减小，从而减少跨边缘平滑造成的几何边界模糊。

对应实现：`edge_aware_smooth_loss()`。

## 6. 多阶段总损失

### 6.1 深度监督

每个阶段在有效像素上使用 Smooth L1：

$$
\mathcal L_{\mathrm{depth}}^s=
\operatorname{SmoothL1}
\left(D_s-D_s^{\mathrm{gt}}\right).
$$

默认阶段权重为

$$
(w_1^D,w_2^D,w_3^D)=(0.5,1.0,2.0).
$$

### 6.2 当前代码对应的总目标

以一基阶段编号 $s\in\{1,2,3\}$ 表示，当前总损失可写为

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

其中 $\mathcal L_{\mathrm{curv}}^s$ 默认是双区域曲率损失；禁用双区域模式时则使用软几何权重曲率损失。

**实现差异提示：**当前代码中法线平滑、边缘感知平滑和深度—法线一致性按 $2^{-(s-1)}$ 衰减；曲率项采用 $s/2$，即三个阶段依次为 $0.5、1.0、1.5$。因此，不能把当前曲率项写成相同的逐阶段二分之一衰减。

### 6.3 默认超参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `dlossw` | $[0.5,1.0,2.0]$ | 三阶段深度监督权重 |
| `normal_smooth_loss_weight` | $0.02$ | 法线平滑基础权重 |
| `curv_loss_weight` | $0.005$ | 曲率连续性基础权重 |
| `edge_smooth_loss_weight` | $0.005$ | 边缘感知平滑基础权重 |
| `depth_normal_loss_weight` | $0.03$ | 深度—法线一致性基础权重 |
| `depth_normal_conf_threshold` | $0.8$ | 高置信硬阈值 |
| `edge_grad_threshold` | $0.05$ | 非边缘硬阈值 |
| `geometry_conf_mid` | $0.65$ | 连续置信权重中点 |
| `geometry_k_conf` | $10.0$ | 置信 sigmoid 斜率 |
| `geometry_edge_mid` | $0.25$ | 连续边缘权重中点 |
| `geometry_k_edge` | $10.0$ | 边缘 sigmoid 斜率 |
| `geometry_w_min` | $0.05$ | 有效像素最小几何权重 |
| `use_dual_region_curvature` | 启用 | 使用 Region A/B 双区域曲率 |
| `region_lambda_a` | $1.5$ | Region A 曲率权重 |
| `region_lambda_b` | $1.0$ | Region B 曲率权重 |
| `region_edge_threshold` | $0.25$ | Sobel 图像边缘阈值 |
| `region_depth_threshold` | $0.02$ | 归一化深度边缘阈值 |
| `region_curv_threshold` | $0.02$ | 高曲率阈值 |
| `region_smooth_k` | $2.0$ | Region B 图像边缘衰减系数 |

## 7. 信息来源、约束与作用关系

| 几何信息 | 直接来源 | 主要参与模块 | 作用 |
|---|---|---|---|
| 预测深度 $D_s$ | 概率体期望回归 | 全部几何约束 | 提供表面位置与形状基础 |
| 多视图重投影 | 相机投影矩阵与深度假设 | 方差代价体 | 建立跨视图对应关系 |
| 概率体置信度 $C_s$ | 回归索引附近四个深度概率之和 | 硬掩码、软权重、Region B | 区分可靠与不可靠深度 |
| 深度反算法线 $N_s^D$ | $D_s$、$K_s$、三维点切向量 | 深度—法线一致性 | 提供显式相机几何法线 |
| 网络预测法线 $N_s^{pred}$ | 参考特征、深度、置信度 | 深度—法线一致性 | 学习表面方向表示 |
| 图像边缘 | 一阶差分或 Sobel | 非边缘掩码、软权重、区域划分、边缘平滑 | 保护外观/结构边界 |
| 深度边缘 | 归一化深度一阶差分 | Region A 划分 | 识别几何不连续 |
| 曲率 | 归一化深度二阶差分 | 区域划分、曲率损失 | 描述局部弯曲和二阶连续性 |

整体关系可概括为：

$$
\text{多视图图像与相机}
\rightarrow
\text{重投影代价体}
\rightarrow
(P_s,D_s,C_s)
\rightarrow
\begin{cases}
N_s^D,\;N_s^{pred},\\
E_s,\;G_s^D,\;\kappa_s
\end{cases}
\rightarrow
\text{掩码/权重/区域}
\rightarrow
\text{几何约束}.
$$

## 8. 公式与代码对应关系

| 内容 | 当前实现位置 |
|---|---|
| 深度概率体 softmax、置信度 | `models/cas_mvsnet.py`：`DepthNet.forward()` |
| 多视图特征重投影 | `models/module.py`：`homo_warping()` |
| 方差代价体 | `models/cas_mvsnet.py`：`DepthNet.forward()` 中的 `volume_variance` |
| 概率期望深度回归 | `models/module.py`：`depth_regression()` |
| 网络预测法线 | `models/module.py`：`NormalHead.forward()` |
| 每阶段法线分支输入与输出 | `models/cas_mvsnet.py`：`CascadeMVSNet.forward()` |
| 相机内参感知的深度反算法线 | `models/module.py`：`compute_normal_from_depth()` |
| 简化深度梯度法线 | `models/module.py`：`get_depth_normals()` |
| 一阶差分图像边缘 | `models/module.py`：`image_gradient_magnitude()` |
| Sobel 图像边缘 | `models/module.py`：`sobel_gradient_magnitude()` |
| 归一化深度边缘 | `models/module.py`：`depth_gradient_magnitude()` |
| 深度曲率图 | `models/module.py`：`curvature_magnitude()` |
| 高置信非边缘硬掩码 | `models/module.py`：`build_smooth_mask()` |
| 连续几何权重 | `models/module.py`：`build_geometry_weight()` |
| Region A/B 构造 | `models/module.py`：`build_dual_region_geometry()` |
| 深度—法线一致性 | `models/module.py`：`depth_normal_consistency_loss()` |
| 法线平滑 | `models/module.py`：`normal_smooth_loss()` |
| 原始曲率损失 | `models/module.py`：`curvature_loss()` |
| 软权重曲率损失 | `models/module.py`：`soft_curvature_loss()` |
| 双区域曲率损失 | `models/module.py`：`dual_region_curvature_loss()` |
| 边缘感知平滑 | `models/module.py`：`edge_aware_smooth_loss()` |
| 多阶段总损失组合 | `models/module.py`：`cas_mvsnet_loss()` |
| 默认权重与阈值 | `train.py` 参数定义及 `loss_kwargs()` |

## 9. 实现边界与使用说明

1. **多视图重投影不是独立损失。** 当前实现用它生成重投影特征和方差代价体，未在 `cas_mvsnet_loss` 中计算额外的光度或几何重投影误差。
2. **置信度不直接接收这些几何权重的梯度。** `DepthNet` 在 `torch.no_grad()` 中生成置信度，`build_geometry_weight()` 和双区域构造又对置信度执行 `detach()`。
3. **区域划分不参与反向传播。** `build_dual_region_geometry()` 使用 `depth.detach()` 计算深度边缘和曲率阈值区域；区域确定后，曲率损失仍作用于未分离的预测深度。
4. **两种深度法线用途不同。** `compute_normal_from_depth()` 使用相机内参和三维切向量，服务于深度—法线一致性；`get_depth_normals()` 使用 $[-D_x,-D_y,1]$ 的简化形式，服务于法线平滑。
5. **NormalHead 没有真值法线监督。** 其主要显式训练信号来自深度—法线一致性，且只在高置信、非边缘、有效区域内计算。
6. **曲率阶段权重以当前代码为准。** 当前实现使用 $0.5、1.0、1.5$，与其他几何项的 $1、1/2、1/4$ 衰减不同。

## 10. 总结

改进网络将多视图重投影产生的深度概率信息进一步转化为深度、置信度、法线、边缘和曲率等几何表示。在此基础上：

- 深度—法线一致性约束局部三维方向；
- 法线平滑抑制表面方向抖动；
- 曲率连续性控制二阶形状变化；
- 边缘感知平滑在去噪的同时保护边界；
- 硬掩码、连续权重和 Region A/B 双区域机制共同决定约束应当施加在何处以及施加多强。

这些机制不改变 CasMVSNet 的多视图代价体、深度假设采样和级联框架，而是在各阶段预测结果上加入显式几何监督与可靠性调制。
