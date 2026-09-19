# 实验15-C：末端激活 A/B 审计结果分析

日期：2026-09-13。研究主体：**SGER-CasMVSNet（自几何与边缘细化残差-CasMVSNet网络）**。

依据现有《实验15-C末端激活AB审计说明.md》，本次直接复核 Epoch9、12、15 的已有 JSON，并核对当前活动源码。原说明保留不变；本次未重新执行 GPU 审计、训练或融合，未修改网络、损失和 checkpoint。

## 1. 主要结论

- **局部机制通过：**Epoch9/12 的全部 10 个配对 Batch 中，仅替换残差头第三个 ReLU，就使残差主体和 output.weight 恢复有限非零梯度及末端特征空间变化，支持该末端激活是这些输入上的局部阻断位置。
- **早期细化尚未恢复：**Epoch9/12 的门控仍全部低于 BCE 的 clamp 下界，收益门控专属梯度为零；两种激活的 refined 与 raw 无误差差异。残差能学习不代表门控能学习，也不代表残差方向正确。
- **直接替换历史权重的质量结论为负：**Epoch15 全部 5 个 Batch 中，两种激活的 refined 都优于各自相同的 raw，但原 ReLU 的 refined 始终更好。LeakyReLU 使汇总细化收益减少约 54.7%。目前不支持将其直接替换为最终推理模型，也不足以判断重新训练后的结果。
- **SGER 本身已有局部收益证据：**Epoch15 ReLU 的 raw→refined 平均误差改善约 0.97%，但这只是 scan2 固定训练样本上的深度证据，尚未达到跨场景点云 Acc./Comp./Overall 的最终贡献标准。

## 2. 数据完整性、配置和统计口径

来源为 `outputs/experiment15_c/activation_ab_15b_epoch{9,12,15}_batch{0000,0001,0002,0003,0004}_{relu,leaky_relu}.json`，共 30 份变体报告、15 份比较报告、450 个 case。Epoch15 的全部五批已经完成，本报告据此汇总，不能沿用仅有 Batch0 时的判断范围。

checkpoint 为 `checkpoints/experiment15_b_backbone_lr1/model_000009.ckpt`、`model_000012.ckpt`、`model_000015.ckpt`。报告中的 checkpoint epoch 与 audit epoch 分别为 9/12/15，沿用文件内的零起始编号；权重 missing/unexpected keys 全部为空。历史训练源码和优化器轨迹未由本次复核重建，不能把当前梯度图自动视为历史训练梯度图。

实际为 train 模式、DTU train split、Batch=2、固定批次 0–4、dataset indices 0–9、seed=1、CUDA/PyTorch 1.13.1+cu117、3 视图、48/32/8 深度假设、4/2/1 区间，Stage3 为 512×640。SGER-lite 仅末级细化，主干未冻结，backbone_lr_scale=0.1。样本仅来自 scan2：参考视角0的光照0–6及参考视角1的光照0–2，包含重复几何；每个 epoch/变体共 2,509,332 个有效像素观测，不能称为同等数量的独立点。

A=ReLU，B=LeakyReLU(negative_slope=0.01)。所有 A/B 配对元数据除实际激活及实际斜率外一致，arguments 中基础激活仍为 relu，以 actual_residual_final_activation 为实际执行配置。当前审计代码每个 case 恢复参数与 BN 缓冲，变体间恢复随机数状态；只加载模型权重，不恢复历史优化器动量、不执行 optimizer.step。

| Epoch | beta | 残差目标有效权重 | 收益门控有效权重 | refined 深度有效权重 |
|---|---:|---:|---:|---:|
| 9 | 0.75 | 0.05 | 0.0375 | 0.75 |
| 12 | 1 | 0.05 | 0.05 | 1 |
| 15 | 1 | 0.05 | 0.05 | 1 |

深度与门控指标均取 full_schedule、Stage3 GT mask>0.5。均值按每批 valid_pixel_count 加权；precision/recall 先汇总计数再相除；不平均分位数。方向准确率还剔除绝对值≤1e-6的残差目标，因此报告逐 Batch 范围，不冒充精确池化准确率。激活统计覆盖全部 B,C,H,W，不用 GT mask。误差沿用数据集原始深度单位。

## 3. 残差主体：早期解除阻断，后期扰动已有提案

以下梯度为 residual_target_only，专属残差目标权重为1且调度强制为1；不能与 full_schedule 的0.05加权梯度混为一谈。主体 L2 是主体全部参数梯度的联合范数，表内为五批最小值–最大值。

| Epoch | 激活 | 主体梯度 L2 | output.weight 梯度 L2 | 平均绝对提案 | 提案方向准确率范围 |
|---|---|---:|---:|---:|---:|
| 9 | relu | 0.000000–0.000000 | 0.000000–0.000000 | 0.001920 | 29.37%–36.90% |
| 9 | leaky_relu | 0.115002–0.142161 | 0.128004–0.158744 | 0.134952 | 29.37%–36.90% |
| 12 | relu | 0.000000–0.000000 | 0.000000–0.000000 | 0.001524 | 45.73%–50.88% |
| 12 | leaky_relu | 0.073520–0.089857 | 0.082238–0.099841 | 0.143630 | 45.73%–50.88% |
| 15 | relu | 0.184479–0.438099 | 0.016019–0.041564 | 0.174094 | 61.34%–63.70% |
| 15 | leaky_relu | 2.942044–3.940363 | 0.118427–0.154283 | 0.235800 | 53.26%–56.01% |

Epoch9/12 的 ReLU 末端输出100%为零、空间方差为0，主体梯度张量存在但值全零，output.weight 也为零；output.bias 仍有非零梯度，因此不是整个残差损失断开。LeakyReLU 末端零值比例降到0，空间方差分别为0.000632–0.000700、0.000704–0.000769，即使输出仍全部为负，学习通路也已经恢复。两组方向准确率逐批相同，说明增大提案和恢复空间变化没有自动纠正方向。

Epoch15 原 ReLU 末端仍有约95.5%–95.7%零值，但约4.3%–4.5%的正值已足以维持主体梯度；不能仅以高零值比例判定主体死亡。其空间方差0.032153–0.039237，也远非早期的全零状态。当前证据不支持“这些通道或整个残差分支在全部训练数据上永久死亡”；后续活跃可能与其他输入、训练更新或优化器历史有关，具体原因需要额外证据。

## 4. 门控：Epoch9/12 存在独立数值阻断

A/B 的 final_gate、logits_a/b、sigmoid_a/b 分布统计逐批完全一致；这是报告统计的一致性，报告未保存逐像素门控差值，不能把它升级为单独测得的逐像素门控相等。

| Epoch | 激活 | 门控均值 | 收益正标签率 | 预测正样本数 | 真阳性数 | gate_benefit_only 门控梯度 L2 |
|---|---|---:|---:|---:|---:|---:|
| 9 | relu | 5.93335e-13 | 0.0000% | 0 | 0 | 0.000000–0.000000 |
| 9 | leaky_relu | 5.93335e-13 | 10.7755% | 0 | 0 | 0.000000–0.000000 |
| 12 | relu | 3.9112e-13 | 0.0000% | 0 | 0 | 0.000000–0.000000 |
| 12 | leaky_relu | 3.9112e-13 | 19.0527% | 0 | 0 | 0.000000–0.000000 |
| 15 | relu | 0.301793 | 21.7079% | 407149 | 156523 | 0.396975–0.861025 |
| 15 | leaky_relu | 0.301793 | 29.8773% | 407149 | 149246 | 3.504340–4.315029 |

Epoch9/12 每组全部2,509,332个有效像素的门控低于1e-6，跨批最大值仅6.42813e-7。LeakyReLU 已分别产生270,393和478,095个正收益标签，但收益门控专属 case 的两分支参数梯度仍为零张量，并非参数未纳入优化器。

当前活动 `models/module.py::balanced_binary_cross_entropy` 先执行 `prediction.clamp(1e-6, 1-1e-6)`，再计算 BCE。门控处于下界外时，该 clamp 对输入的导数为零。这与“非零正标签、有效专属权重、门控专属梯度全零”共同支持当前门控损失通路的数值阻断；不能据此单独解释门控历史上最初为何饱和。

早期 ReLU 无收益正标签且 BCE 约1e-6；LeakyReLU 的 BCE 约6.907756，但两者门控梯度都为零。损失数值大不保证存在可学习梯度。早期 precision 在预测正样本为0时无统计定义，ReLU 的 recall 也因目标正样本为0而无定义；原日志通过分母加 epsilon 输出0，不能直接解释为有效分类准确率。

Epoch15 门控均值约0.301793，没有有效像素落在1e-6下界外，收益门控梯度恢复。ReLU 的汇总 precision/recall 为38.44%/28.73%，LeakyReLU 为36.66%/19.91%。两组门控输出统计相同，但提案改变了收益标签，因此分类指标的变化不等于门控参数发生更新。

Epoch15 ReLU 在 Batch1–3 的“恶化像素门控均值”高于“改善像素门控均值”；Batch0/4相反。门控选择能力仍有改善空间，不过这些条件均值不能单独证明关门必定降低总误差，亦不能将 gate>=0.5 当作无阈值加权细化的全部机制。

## 5. raw/refined：Epoch15 五批一致支持原 ReLU

Δ=raw误差−refined误差，正值为改善。下表为有效像素加权汇总。

| Epoch | raw | ReLU refined | LeakyReLU refined | ReLU Δ | LeakyReLU Δ |
|---|---:|---:|---:|---:|---:|
| 9 | 2.141281 | 2.141281 | 2.141281 | 0.000000 | 0.000000 |
| 12 | 2.004204 | 2.004204 | 2.004204 | 0.000000 | 0.000000 |
| 15 | 2.004833 | 1.985333 | 1.995999 | 0.019500 | 0.008834 |

Epoch9/12 LeakyReLU 平均绝对有效残差仅约1.37e-14/1.20e-14；误差改善和恶化像素比例均为0。门控使提案在 float32 深度加法中无法体现为有效修正，不能将“refined未退化”解读为提案安全。

Epoch15 逐批证据：

| Batch | raw | ReLU refined | LeakyReLU refined | LeakyReLU相对ReLU误差增加 |
|---|---:|---:|---:|---:|
| 0 | 2.509561 | 2.485546 | 2.499053 | +0.013507 |
| 1 | 2.509062 | 2.493209 | 2.503223 | +0.010014 |
| 2 | 2.530754 | 2.517298 | 2.526521 | +0.009222 |
| 3 | 1.688880 | 1.670016 | 1.678522 | +0.008506 |
| 4 | 0.912048 | 0.887242 | 0.899290 | +0.012048 |

Epoch15 LeakyReLU 的汇总 refined 误差增加0.010666；原有0.019500收益降为0.008834，减少54.7%。改善像素比例从60.54%降到51.82%，恶化比例从39.38%增到48.14%；平均绝对有效残差反而从0.065222增到0.073493。即修正更大、整体收益更小。

未门控提案收益定义为 `abs(raw-GT)-abs(raw+proposal-GT)`，未乘门控和beta：

| Epoch | ReLU平均提案收益 | LeakyReLU平均提案收益 |
|---|---:|---:|
| 9 | -0.000654 | -0.063450 |
| 12 | -0.000032 | -0.028803 |
| 15 | 0.036179 | -0.015011 |

Epoch15 ReLU 五批提案收益全部为正，LeakyReLU 五批全部为负。后者收益标签率仍更高，是因为超过固定margin的局部正收益像素增多；同时负收益的幅度和分布也改变。因此正标签率增加不能证明平均方向和质量变好。固定门控和幅度调制下的最终 LeakyReLU refined 仍有小幅正收益，但没有提升原模型。

## 6. 审计边界、后续次序与复核记录

本次核验15个A/B比较的 raw depth、raw photometric confidence 最大绝对差均严格为0；450个case的参数梯度汇总均有限，优化器参数覆盖完整，提案与有效残差有限且不超过记录上界（复核容差1e-6）。240个SGER专属case的主干梯度范数均为0。对早期零损失/零梯度case，单凭零主干梯度不足以证明有效隔离；LeakyReLU已恢复的非零残差目标梯度及Epoch15活跃case补充了这一验证。

Epoch9/12共20份报告均只有同一fatal error：`gate_benefit_only: missing critical SGER benefit gate gradient`；Epoch15共10份报告fatal errors为空。早期输出完整但审计未整体通过，非零退出码不能解释为没有生成结果。该错误中的“missing”在这里具体是已有梯度张量但全部为零。

另外，SGER-lite 当前仅调用末级 normal head。报告把前两级normal head归入other，其16个参数张量无梯度，但优化器覆盖完整。这与当前 `run_normal_head` 条件一致，应区别于末级SGER目标分支失去学习信号；不能宣称模型所有参数都有非零梯度。

建议按单变量次序推进：

1. **先补覆盖和历史定位。**在其他scan/参考视角增加配对样本，围绕Epoch12→15检查中间checkpoint和既有训练日志，定位残差与门控恢复时点。分别记录末端输入正值比例、主体梯度、输出偏置、门控下界比例、提案方向和raw/refined误差。若缺少历史优化器状态或源码，应明确无法归因，不能从三个快照推断完整训练过程。
2. **激活改动作为受控短程训练候选。**它针对SGER残差主体的负区间梯度阻断。固定门控、BCE、margin、损失、种子、数据顺序和优化器恢复策略，仅对比末端ReLU/LeakyReLU。既观察早期通路恢复，也保留Epoch15已活跃模型的副作用对照。重新训练的判据是方向和raw/refined收益，不能只看梯度更大。当前无需据此直接启动完整训练。
3. **收益门控数值修正独立评估。**它针对SGER门控在极低概率下不能接收正收益监督的问题。固定选定激活，验证双区域门控及几何权重组合后的最终收益概率，在极端负logits和正标签下能得到有限且方向正确的梯度。任何分支logit都不能直接冒充混合门控的最终logit；保持margin和损失权重不变，避免与激活改动混在同轮归因。数值修复不等于应无差别开门，早期提案平均收益为负尤其需要选择性验证。
4. **最终以SGER独立贡献收尾。**同checkpoint、同raw confidence、相同场景/视图/分辨率/深度采样和融合阈值，比较raw与refined的逐场景及汇总Acc./Comp./Overall，另查跨checkpoint或重复实验稳定性。当前未采集点云指标、点数、融合掩码通过率及一致性计数，不能宣称最终重建指标已提升。

汇总及来源SHA256：`outputs/experiment15_c/activation_ab_verified_summary_20260913.json`，包含逐批指标、所有45份输入哈希、当前源码哈希及完整arguments。复核脚本：`outputs/experiment15_c/summarize_activation_ab_20260913.py`。本次执行该脚本完成JSON断言和汇总，没有重新运行神经网络；也没有把历史测试通过记成本次测试。

### 根据报告arguments重建的复核命令（本次未执行）

下列命令以独立`_recheck_20260913`前缀保存新结果。原始shell逐字命令不可得，布尔false参数由解析器默认值保留；use_dual_region_curvature=True也是解析器默认值，命令行仅提供反向关闭开关，故不输出该内部字段。其余参数来自报告。

```bash
OMP_NUM_THREADS=1 python -u gradient_audit.py \
  --activation_ab \
  --audit_epoch 9 \
  --backbone_lr_scale 0.1 \
  --batch_indices 0,1,2,3,4 \
  --batch_size 2 \
  --benefit_margin_ratio 0.05 \
  --cr_base_chs 8,8,8 \
  --curv_loss_weight 0.005 \
  --dataset dtu_yao \
  --depth_inter_r 4,2,1 \
  --depth_normal_conf_threshold 0.8 \
  --depth_normal_loss_weight 0.03 \
  --device cuda \
  --dlossw 0.5,1.0,2.0 \
  --edge_grad_threshold 0.05 \
  --edge_smooth_loss_weight 0.005 \
  --freeze_backbone_epochs 0 \
  --gate_benefit_loss_weight 0.05 \
  --gate_loss_weight 0.0 \
  --geometry_conf_mid 0.65 \
  --geometry_edge_mid 0.25 \
  --geometry_k_conf 10.0 \
  --geometry_k_edge 10.0 \
  --geometry_w_min 0.05 \
  --grad_method detach \
  --interval_scale 1.06 \
  --loadckpt ./checkpoints/experiment15_b_backbone_lr1/model_000009.ckpt \
  --lr 0.001 \
  --ndepths 48,32,8 \
  --normal_smooth_loss_weight 0.02 \
  --num_workers 0 \
  --numdepth 192 \
  --nviews 3 \
  --output_json outputs/experiment15_c/activation_ab_15b_epoch9_recheck_20260913.json \
  --raw_depth_loss_weight 1.0 \
  --refined_depth_loss_weight 1.0 \
  --region_curv_threshold 0.02 \
  --region_depth_threshold 0.02 \
  --region_edge_threshold 0.25 \
  --region_lambda_a 1.5 \
  --region_lambda_b 1.0 \
  --region_smooth_k 2.0 \
  --residual_loss_weight 0.01 \
  --residual_target_loss_weight 0.05 \
  --residual_target_ratio 0.25 \
  --safe_refine_loss_weight 0.1 \
  --safe_refine_margin 0.0 \
  --seed 1 \
  --sger_feature_channels 8 \
  --sger_gate_channels 16 \
  --sger_hidden_channels 32 \
  --sger_max_residual_ratio 0.25 \
  --sger_residual_final_activation relu \
  --sger_residual_negative_slope 0.01 \
  --sger_warmup_end_epoch 10 \
  --sger_warmup_start_epoch 7 \
  --trainlist lists/dtu/train.txt \
  --trainpath ./data/DTU/mvs_training/dtu/ \
  --use_sger_lite \
  --wd 0.0

OMP_NUM_THREADS=1 python -u gradient_audit.py \
  --activation_ab \
  --audit_epoch 12 \
  --backbone_lr_scale 0.1 \
  --batch_indices 0,1,2,3,4 \
  --batch_size 2 \
  --benefit_margin_ratio 0.05 \
  --cr_base_chs 8,8,8 \
  --curv_loss_weight 0.005 \
  --dataset dtu_yao \
  --depth_inter_r 4,2,1 \
  --depth_normal_conf_threshold 0.8 \
  --depth_normal_loss_weight 0.03 \
  --device cuda \
  --dlossw 0.5,1.0,2.0 \
  --edge_grad_threshold 0.05 \
  --edge_smooth_loss_weight 0.005 \
  --freeze_backbone_epochs 0 \
  --gate_benefit_loss_weight 0.05 \
  --gate_loss_weight 0.0 \
  --geometry_conf_mid 0.65 \
  --geometry_edge_mid 0.25 \
  --geometry_k_conf 10.0 \
  --geometry_k_edge 10.0 \
  --geometry_w_min 0.05 \
  --grad_method detach \
  --interval_scale 1.06 \
  --loadckpt ./checkpoints/experiment15_b_backbone_lr1/model_000012.ckpt \
  --lr 0.001 \
  --ndepths 48,32,8 \
  --normal_smooth_loss_weight 0.02 \
  --num_workers 0 \
  --numdepth 192 \
  --nviews 3 \
  --output_json outputs/experiment15_c/activation_ab_15b_epoch12_recheck_20260913.json \
  --raw_depth_loss_weight 1.0 \
  --refined_depth_loss_weight 1.0 \
  --region_curv_threshold 0.02 \
  --region_depth_threshold 0.02 \
  --region_edge_threshold 0.25 \
  --region_lambda_a 1.5 \
  --region_lambda_b 1.0 \
  --region_smooth_k 2.0 \
  --residual_loss_weight 0.01 \
  --residual_target_loss_weight 0.05 \
  --residual_target_ratio 0.25 \
  --safe_refine_loss_weight 0.1 \
  --safe_refine_margin 0.0 \
  --seed 1 \
  --sger_feature_channels 8 \
  --sger_gate_channels 16 \
  --sger_hidden_channels 32 \
  --sger_max_residual_ratio 0.25 \
  --sger_residual_final_activation relu \
  --sger_residual_negative_slope 0.01 \
  --sger_warmup_end_epoch 10 \
  --sger_warmup_start_epoch 7 \
  --trainlist lists/dtu/train.txt \
  --trainpath ./data/DTU/mvs_training/dtu/ \
  --use_sger_lite \
  --wd 0.0

OMP_NUM_THREADS=1 python -u gradient_audit.py \
  --activation_ab \
  --audit_epoch 15 \
  --backbone_lr_scale 0.1 \
  --batch_indices 0,1,2,3,4 \
  --batch_size 2 \
  --benefit_margin_ratio 0.05 \
  --cr_base_chs 8,8,8 \
  --curv_loss_weight 0.005 \
  --dataset dtu_yao \
  --depth_inter_r 4,2,1 \
  --depth_normal_conf_threshold 0.8 \
  --depth_normal_loss_weight 0.03 \
  --device cuda \
  --dlossw 0.5,1.0,2.0 \
  --edge_grad_threshold 0.05 \
  --edge_smooth_loss_weight 0.005 \
  --freeze_backbone_epochs 0 \
  --gate_benefit_loss_weight 0.05 \
  --gate_loss_weight 0.0 \
  --geometry_conf_mid 0.65 \
  --geometry_edge_mid 0.25 \
  --geometry_k_conf 10.0 \
  --geometry_k_edge 10.0 \
  --geometry_w_min 0.05 \
  --grad_method detach \
  --interval_scale 1.06 \
  --loadckpt ./checkpoints/experiment15_b_backbone_lr1/model_000015.ckpt \
  --lr 0.001 \
  --ndepths 48,32,8 \
  --normal_smooth_loss_weight 0.02 \
  --num_workers 0 \
  --numdepth 192 \
  --nviews 3 \
  --output_json outputs/experiment15_c/activation_ab_15b_epoch15_recheck_20260913.json \
  --raw_depth_loss_weight 1.0 \
  --refined_depth_loss_weight 1.0 \
  --region_curv_threshold 0.02 \
  --region_depth_threshold 0.02 \
  --region_edge_threshold 0.25 \
  --region_lambda_a 1.5 \
  --region_lambda_b 1.0 \
  --region_smooth_k 2.0 \
  --residual_loss_weight 0.01 \
  --residual_target_loss_weight 0.05 \
  --residual_target_ratio 0.25 \
  --safe_refine_loss_weight 0.1 \
  --safe_refine_margin 0.0 \
  --seed 1 \
  --sger_feature_channels 8 \
  --sger_gate_channels 16 \
  --sger_hidden_channels 32 \
  --sger_max_residual_ratio 0.25 \
  --sger_residual_final_activation relu \
  --sger_residual_negative_slope 0.01 \
  --sger_warmup_end_epoch 10 \
  --sger_warmup_start_epoch 7 \
  --trainlist lists/dtu/train.txt \
  --trainpath ./data/DTU/mvs_training/dtu/ \
  --use_sger_lite \
  --wd 0.0

```
