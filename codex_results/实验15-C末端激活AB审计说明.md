# 实验15-C末端激活A/B审计

2026-09-11。SGER-CasMVSNet（自几何与边缘细化残差-CasMVSNet网络）。针对已观察到的ResidualDepthHead第三个ReLU全负截断，只改变body.5末端激活；前两个ReLU、门控、BCE、margin、损失权重和优化调度保持不变。

## 配置

模型、train.py、test.py、gradient_audit.py均支持：

- `--sger_residual_final_activation relu`（默认）或`leaky_relu`。
- `--sger_residual_negative_slope 0.01`，有限且严格位于(0,1)，仅LeakyReLU实际使用。

只替换无参数激活，state_dict键和参数形状不变，可严格加载旧权重。新训练checkpoint记录上述配置；恢复训练和推理时仍须显式传入一致的命令行参数，加载权重不会自动恢复激活类型。SGER-Lite只作用于最终Stage；若使用全阶段SGER，该开关应用于各SGERBlock的末端激活，不代表此次单阶段实验。

## 同checkpoint、同Batch A/B

```bash
OMP_NUM_THREADS=1 python -u gradient_audit.py \
  --trainpath ./data/DTU/mvs_training/dtu/ \
  --trainlist lists/dtu/train.txt \
  --loadckpt ./checkpoints/experiment15_b_backbone_lr1/model_000009.ckpt \
  --audit_epoch 9 --batch_size 2 --seed 1 \
  --batch_indices 0,1,2,3,4 \
  --freeze_backbone_epochs 0 --backbone_lr_scale 0.1 \
  --activation_ab --sger_residual_negative_slope 0.01 \
  --output_json outputs/experiment15_c/activation_ab_15b_epoch9.json
```

Epoch12/15分别替换checkpoint、audit_epoch和输出名称。需要逐层进度可加`--verbose_activation_progress`。补充跨场景样本时指定经过dataset.metas核对的Batch编号；前5批仅覆盖scan2的两个参考视角。

每个Batch生成三个文件：

- `activation_ab_15b_epoch9_batch0000_relu.json`
- `activation_ab_15b_epoch9_batch0000_leaky_relu.json`
- `activation_ab_15b_epoch9_batch0000_ab_comparison.json`

两组共享同一次加载的权重、同一已加载Batch，每组及各case恢复模型参数与BN；两组恢复相同PyTorch CPU/CUDA随机数状态。审计保持train模式，不执行optimizer.step，不改写checkpoint。A/B结束或异常时恢复原激活。`--activation_ab`固定依次运行relu/leaky_relu，覆盖单组激活选项；actual_residual_final_activation/actual_residual_negative_slope明确记录实际配置。

每组含原15个case和原有残差/门控统计。新增`loss_extras.residual_bound_checks`记录全图提案/有效残差最大绝对值、理论上界和有限性；有效上界包含case的beta。比较文件包含两组full_schedule误差、residual_target_only梯度汇总、错误列表与raw深度/raw confidence逐像素最大绝对差，应为0。原版门控梯度缺失时仍继续修改版并输出比较，最终退出码可能为1，须读取错误内容而非把非零退出码视作程序未完成。

## 判断与验证

首先检查主体和output.weight有限非零梯度、末端激活/提案空间变化、残差上界及主干隔离；然后检查提案方向与raw/refined误差。非零梯度不等于重建收益；直接用历史权重替换激活可能立即改变提案幅度甚至恶化深度，不应据此直接完整训练。当前门控硬clamp保留，即使提案恢复，也可能继续阻断门控学习。

专项测试`PYTHONPATH=.:tests OMP_NUM_THREADS=1 python tests/test_activation_ab.py`验证可控末端失活时主体梯度恢复、前两激活不变、参数与state_dict键不变、非法斜率拒绝，以及完整生产CascadeMVSNet/损失的CPU合成A/B入口、两组报告、raw不变、残差上界与隔离。合成输入为32×32、Batch=2、8/8/8采样；不能代替真实DTU审计。此次未启动真实checkpoint A/B或完整训练。
