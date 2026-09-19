# 实验15-C：已训练 checkpoint 单 Batch 审计分析

日期：2026-09-08。研究主体：SGER-CasMVSNet（自几何与边缘细化残差-CasMVSNet网络）。本次只分析报告和活动源码，未修改模型、损失或 checkpoint，未执行新训练。

## 数据与口径

来源：outputs/experiment15_c/audit_15b_epoch9.json、audit_15b_epoch12.json、audit_15b_epoch15.json；audit_epoch10.json 为随机初始化对照，不能作为训练后 Epoch10。

三个模型来自 checkpoints/experiment15_b_backbone_lr1/model_000009.ckpt、model_000012.ckpt、model_000015.ckpt。报告内 checkpoint_epoch 与 audit_epoch 分别一致，missing/unexpected keys 均为空。统一 DTU train split、首 Batch、shuffle=False、batch_size=2、seed=1、3视图、48/32/8深度采样、4/2/1区间倍率、0.5/1/2深度损失权重、SGER-Lite、CUDA/PyTorch 1.13.1+cu117。Stage3 有效像素均为487128。运行脚本使用 train() 模式，结果不是验证集或DTU融合评估。

主干未冻结，各组参数梯度均有限，本次构建的优化器无参数遗漏。backbone_lr_scale=0.1 是审计优化器配置；审计不执行 step，因此它不会缩放 backward 梯度，也不代表历史15-B使用该学习率。

## Stage3前向结果（full_schedule，有效GT掩码内像素均值）

| checkpoint epoch | 平均绝对提案 | 门控均值 | 平均绝对有效残差 | raw误差 | refined误差 | 残差方向准确率 | 收益目标正样本率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9 | 0.00192001 | 1.50079e-13 | 2.16115e-16 | 2.652235 | 2.652235 | 29.37% | 0.00% |
| 12 | 0.00152416 | 3.91087e-13 | 5.96079e-16 | 2.493480 | 2.493480 | 45.73% | 0.00% |
| 15 | 0.176208 | 0.297341 | 0.0654377 | 2.509561 | 2.485546 | 63.21% | 23.50% |

Epoch9 beta=0.75，Epoch12/15 beta=1；残差目标有效权重始终为0.05，收益门控有效权重分别为0.0375、0.05、0.05。Epoch9 full_forced 强制beta和门控调度为1也未恢复细化，因此不能归因于warm-up主动关闭。

## 主要诊断

1. Epoch9/12存在数值上极端关闭的最终门控。提案仍非零，但只有约0.0015–0.0019，较Epoch15的0.1762小约92–116倍；同时方向准确率低。有效残差降至1e-16量级的直接抑制因素是1e-13量级门控，不能把提案称为已经正常，也不能断言SGER所有分支都已失去学习能力。
2. 两个checkpoint的收益目标正样本率均为0，precision/recall为0不是单纯的评估阈值假象。目标函数要求未门控提案减少误差超过0.05倍depth_interval；方向正确不保证超过该收益门槛。当前没有逐像素提案和阈值分布，不能量化无正样本具体由幅度还是方向主导。
3. gate_benefit_only 的损失均为1.01328e-6，但所有SGER参数梯度为0；两个JSON均记录 fatal_errors: missing critical SGER benefit gate gradient。这是门控学习信号诊断失败，不是审计程序未执行，也不是主干泄漏。
4. 活动 models/module.py:876 的 balanced_binary_cross_entropy 对输入先做 clamp(1e-6,1-1e-6)。区间外 clamp 梯度为0，因而产生硬截断区间。观测到极低门控、相同损失下限和门控参数零梯度，与该机制一致。最终门控还包含几何权重，不是单个sigmoid；应进一步采集分支logit、sigmoid和几何权重的分布，才能定位各分支饱和程度。
5. refined_only 的SGER梯度在Epoch9/12仅5.11e-14、1.83e-13，训练信号几乎消失；但 residual_target_only 仍有0.1303、0.02538梯度。这说明残差目标监督仍能驱动提案，不支持“整个SGER彻底无法更新”的结论。
6. “提案不足→收益标签全负→门控关闭→refined监督衰减”是与证据一致的优化反馈假设。仅三个checkpoint和一个Batch不足以证明发生顺序，也不能解释Epoch15为何恢复；其他Batch、NormalHead输入变化和历史优化器动量均未审计。

## 梯度边界

三个checkpoint中 raw_only 的Backbone范数为2.89590、1.70975、1.90197，SGER为0。四个SGER专属损失的Backbone梯度均为0。

Epoch9/12 safe-refine损失未激活，不能据其零梯度验证连接。Epoch15提供了有效验证：refined_only、residual_target_only、gate_benefit_only、safe_refine_only 的SGER范数分别为1.50135、0.44582、0.48897、1.02512，Backbone全部为0；safe-refine损失为0.0189512。

因此当前源码在这个Batch及已激活损失下符合核心隔离预期。不能由此证明或否定历史15-B代码泄漏；当前源码已经有SGER-Lite detach边界，加载历史权重不会恢复历史计算图。几何平滑审计汇总所有stage，仍须单独审计Stage3才能完整确认SGER几何边界。

## Epoch15恢复后的质量

raw误差2.509561→refined误差2.485546，降低0.024015，约0.96%；改善像素61.25%，恶化38.67%。这是该训练Batch的正收益，不是DTU Acc./Comp./Overall贡献证明。

方向准确率63.21%低于实验目标70%；gate precision=42.94%低于60%，recall=29.08%。改善像素门控均值0.297507、恶化像素0.297380，差值仅0.000127，远低于目标0.20。该均值差说明对实际细化收益的分离较弱，但不能断言门控毫无分类能力：precision高于收益目标正样本率23.50%，且门控分类目标与实际细化改善条件不同。

normal_depth_cos在Epoch9/12/15分别约0.98883、0.99439、0.68093；本Batch的Epoch9并未复现历史汇总中约0.42的数值，不能混用不同聚合范围。高法线自洽并不能阻止门控关闭，也不能证明深度准确。

## 下一步（建议，未实施）

优先完善SGER门控数值诊断：按stage和分支采集logit、sigmoid、几何权重、最终门控分位数、低于1e-6比例、预测正样本/目标正样本/真阳性计数，以及提案收益与margin分布。用固定的多个Batch检查，并分别注明train/eval模式；补Stage3几何损失审计。

若这些检查确认硬clamp截断是门控优化障碍，下一轮将“消除收益门控损失的硬截断梯度区间”作为唯一主要因素，先以饱和输入和正/负标签做梯度验证，再短程对照。由于当前收益门控是双分支概率与几何权重的组合，不能直接把最终概率当作logit送入BCEWithLogitsLoss，也不应未经推导改监督对象。

不要同时提高Residual Target权重、加Gate预算、缩小残差范围或改变融合阈值。全负标签时稳定BCE也不会自动鼓励开门；数值修正不能替代提案质量和标签分布诊断。最终仍需同checkpoint、同raw confidence的raw/refined对照，以DTU三项指标验证SGER独立贡献。


下一步应先做一轮门控专项诊断，完成后再决定实验15-C的唯一修改项，暂不开始完整训练。
                                                                                                                                                                                                                              
  1. 补齐审计指标。 分别记录 Gate A/B 的 logits、sigmoid 输出、几何权重、最终门控，以及：                                                                                                                                     
      - 最终门控低于 1e-6 的比例；                                                                                                                                                                                            
      - 提案实际收益与 benefit_margin 的分布；                                                                                                                                                                                
      - 收益目标正样本数、预测正样本数和真阳性数；                                                                                                                                                                            
      - Stage3 几何损失单独反传的梯度。                                                                                                                                                                                       
                                                                                                                                                                                                                              
  2. 用 Epoch9、12、15，在相同的5个固定 Batch 上复核。 不更新参数。重点确认：零梯度是否普遍来自 BCE 前的硬 clamp；收益标签全负主要是提案幅度不足，还是方向错误。                                                              
  3. 根据结果只改一个因素。                                                                                                                                                                                                   
      - 若确认硬 clamp 阻断门控学习，优先修正收益门控损失的数值实现，保持现有监督含义。                                                                                                                                       
      - 若数值路径正常，但收益标签持续全负，则先改善残差提案，不能靠强行开门解决。                                                                                                                                            
                                                                                                                                                                                                                              
  4. 通过专项梯度测试后再短程对照训练。 保持学习率、残差范围、其他损失权重和融合阈值固定，比较修改前后门控是否仍饱和、提案方向是否改善，以及 refined 是否稳定优于 raw。                                                       
                                                                                                                                                                                                                              
  当前最值得先做的是第1、2步：补充门控诊断并批量审计。 现有证据还不足以直接选择“提高残差权重”或“加入 Gate 预算”。 