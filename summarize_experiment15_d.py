"""Write a provenance-linked 15-D report from completed audit/probe outputs."""
import json
from pathlib import Path
from datetime import datetime
from experiment15_d import ROOT, SOURCE, sha


def main():
    manifest=json.loads((ROOT/'run_manifest.json').read_text())
    hashes={p:sha(p) for p in SOURCE}
    pairs={}
    for run in manifest['runs']:
        if run['status']!='verified' or run['source_sha256']!=hashes:continue
        for p in run['pairs']:
            assert all(sha(f)==h for f,h in p['files_sha256'].items())
            pairs[(p['epoch'],p['activation'],p['batch'])]=p
    full=(ROOT/'audit_summary.json').exists()
    if full:
        summary=json.loads((ROOT/'audit_summary.json').read_text())
        assert summary['passed'] and len(pairs)==25
    probe_path=ROOT/'short_train/comparison.json'
    probe_manifest_path=ROOT/'short_train/sample_manifest.json'
    probe_manifest=json.loads(probe_manifest_path.read_text()) if probe_manifest_path.exists() else {}
    interrupted=probe_manifest.get('status')=='interrupted_by_user'
    now=datetime.now().strftime('%Y-%m-%d %H:%M')
    text='''# 实验15-D：门控BCE数值稳定性A/B审计结果分析

研究主体：**SGER-CasMVSNet（自几何与边缘细化残差-CasMVSNet网络）**。

'''
    text+='更新：'+now+'。状态：'+('25个配对机制审计已完成并通过。' if full else '最终版本审计进行中，目前已验证%d/25个配对。'%len(pairs))
    if interrupted:
        text+='按用户要求，自动短程训练已中止，后续审计由用户自行执行。部分训练记录保留，不能宣称取得训练或重建精度提升。\n'
    else:
        text+=('两组200步局部训练探针已完成。\n' if probe_path.exists() else '短程训练结果尚未生成，不能宣称取得训练或重建精度提升。\n')
    text+='''
本报告对应[实验方案](实验15-D门控BCE数值稳定性AB实验方案.md)。唯一主要变量是旧的概率clamp BCE与稳定BCE；同一个配对内固定激活、模型参数、输入、BN、随机数、标签及实际有效损失权重。

## 1. 实现与合成验证

活动模型继续使用原双区域前向门控。利用其区域互斥性，把每像素概率表示为p=w·sigmoid(z)，稳定损失从原始logit和原几何权重计算：

- 正标签：softplus(-z)−log(w)。
- 负标签且w=1：softplus(z)。
- 负标签且0<w<1：−log1p(−exp(log(w)+logsigmoid(z)))。
- w=0的负标签损失为0；w=0的正标签显式报告结构不可达，不暗改标签、掩码或几何权重。

保留旧类别平衡、单类回退和归一化。训练默认模式仍为legacy_clamp；stable_logprob必须显式选择，checkpoint保存该配置。原前向门控、残差上界和SGER-lite隔离边界保持不变。

CPU/CUDA分别完成110条数值记录（包括float32/float64正常区间、极端logit、有限差分、空掩码、结构零和微小非负损失验证）；CPU另完成完整生产CascadeMVSNet/损失入口的30个合成case。float32比较原损失，float64使用独立参考表达式，因为旧函数硬转换target.float()，不能直接运行float64 BCE。采用方案中预设容差。

| 设备 | 正常区间最大损失绝对差 | 正常区间最大logit梯度绝对差 |
|---|---:|---:|
'''
    for device in ('cpu','cuda'):
        d=json.loads((ROOT/'synthetic'/('%s.json'%device)).read_text())
        rs=[r for r in d['tests'] if r['test']=='normal_equivalence']
        text+='| %s | %.6g | %.6g |\n'%(device,max(r['loss_abs_diff'] for r in rs),max(r['gradient_max_abs_diff'] for r in rs))
    text+='''
原有激活A/B、门控审计、梯度审计及损失隔离回归已通过；训练CLI到损失模式的参数传递已核验。增加gate_logits_a/b是为了稳定损失及梯度审计，未增加可训练参数；旧checkpoint可以严格加载。

初版logsumexp补概率在极低概率负标签上出现约1e-10负损失舍入误差，已修正并重新运行最终版本试点。初版结果保存在outputs/experiment15_d/pre_revision_logsumexp，不并入本报告最终配对。该修正没有改变数学目标。

## 2. 已验证的真实审计

采用Epoch9/12/15及固定Batch0–4。主要ReLU组覆盖三个epoch；固定LeakyReLU(0.01)压力组覆盖Epoch9/12。模型train模式，SGER-lite仅末级，DTU训练样本scan2、两个参考视角和重复光照，Batch=2、3视图、Stage3 512×640、48/32/8采样、4/2/1区间、seed=1。此审计不执行optimizer.step，也不改写输入checkpoint。

下面梯度取gate_benefit_only（权重1、强制调度1）；不是full_schedule实际0.0375/0.05加权后的梯度。表中范围为已完成批次的最小值–最大值。

| Epoch | 固定激活 | 已验证批数 | 正标签率 | 旧BCE门控参数L2 | 稳定BCE门控参数L2 |
|---|---|---:|---:|---:|---:|
'''
    aggregated=[]
    for e,a in ((9,'relu'),(12,'relu'),(15,'relu'),(9,'leaky_relu'),(12,'leaky_relu')):
        ps=[p for (ep,act,b),p in pairs.items() if ep==e and act==a]
        if not ps:continue
        n=sum(p['variants']['stable_logprob']['logit_directions']['valid_count'] for p in ps)
        pos=sum(p['variants']['stable_logprob']['logit_directions']['positive_count'] for p in ps)
        old=[p['variants']['legacy_clamp']['gate_grad_l2'] for p in ps]
        new=[p['variants']['stable_logprob']['gate_grad_l2'] for p in ps]
        text+='| %d | %s | %d | %.4f%% | %.6g–%.6g | %.6g–%.6g |\n'%(e,a,len(ps),pos/n*100,min(old),max(old),min(new),max(new))
        aggregated.append({'epoch':e,'activation':a,'batches':len(ps),'valid_count':n,'positive_count':pos,
                           'legacy_gate_grad_l2_range':[min(old),max(old)],'stable_gate_grad_l2_range':[min(new),max(new)]})
    text+='\n当前已核验%d个case。'% (len(pairs)*30)
    if pairs:
        text+='''所有已验证配对的每个case，raw深度、raw置信度、提案、最终门控、有效残差、refined深度、几何权重、GT掩码及收益标签逐像素差均为0；参数梯度有限、优化器覆盖完整、残差有限且满足上界。SGER专属case主干梯度均为0，gate_benefit_only不更新残差头。稳定BCE已验证报告的fatal errors为空。

固定LeakyReLU压力组中，可学习正标签的活动logit梯度全部方向正确并到达门控参数；负标签的极小梯度允许在float32下下溢，不能因此判定失败。ReLU早期没有收益正标签，其极小门控梯度不应解读为需要开门。所有已验证case均无结构不可达正标签。

稳定BCE直接连接logits；最终benefit_gate张量上的梯度可为零，因为它是同源前向分支。判断新损失是否可学习使用logit和参数梯度，不能误用旧的gate张量梯度指标。

由于只改损失且不更新参数，本次审计各配对的raw/refined深度误差保持不变。恢复梯度证明的是数值机制，不是即时精度提升。正标签压力组的专属梯度可能较大，必须结合实际损失权重和学习率评估训练，不能同时擅自更改多项超参数。
'''
    text+='''
## 3. 局部训练探针

训练准备核验了Epoch15 checkpoint的优化器状态：SGER和backbone组各25/140个参数张量，两个实际lr均为0.000125，149个参数张量有优化器状态。恢复采用现有分组/顺序并逐项校验动量张量形状和值；历史checkpoint没有参数名清单，不能将这一核验说成已重建历史训练源码。

两组使用相同恢复状态和ReLU；固定Epoch15的beta及有效权重，固定恢复的lr，不进行额外调度。每组200步，预先固定相同的400个训练样本索引，在0/50/100/200步保存快照。验证为scan3、scan5、scan17，参考视角0/1、光照0，dataset indices 0/7/343/350/686/693，已实际加载核验；按有效像素汇总，验证使用eval模式。
'''
    if probe_path.exists():
        probe=json.loads(probe_path.read_text());assert probe['completed_steps_per_variant']==200
        text+='\n| 步数 | 旧BCE raw | 旧BCE refined | 稳定BCE raw | 稳定BCE refined | 旧Δ | 新Δ |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for step in ('0','50','100','200'):
            a=probe['validation'][step]['legacy_clamp'];b=probe['validation'][step]['stable_logprob']
            text+='| %s | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f |\n'%(step,a['raw_abs_error'],a['refined_abs_error'],b['raw_abs_error'],b['refined_abs_error'],a['raw_to_refined_error_delta'],b['raw_to_refined_error_delta'])
        text+='\n训练raw/置信度逐步哈希完全一致：%s；验证raw/置信度哈希完全一致：%s。\n'%(probe['raw_training_trajectory_exact'],probe['raw_validation_exact'])
        text+='\n| 验证scan（200步） | 旧raw | 旧refined | 新raw | 新refined |\n|---|---:|---:|---:|---:|\n'
        ra=json.loads((ROOT/'short_train/legacy_clamp/metrics.json').read_text())['validation']['200']
        rb=json.loads((ROOT/'short_train/stable_logprob/metrics.json').read_text())['validation']['200']
        for scan,a,b in zip(('scan3','scan5','scan17'),ra,rb):
            text+='| %s | %.6f | %.6f | %.6f | %.6f |\n'%(scan,a['raw_abs_error'],a['refined_abs_error'],b['raw_abs_error'],b['refined_abs_error'])
        a=probe['validation']['200']['legacy_clamp'];b=probe['validation']['200']['stable_logprob']
        text+='\n200步后，稳定BCE的汇总refined误差相对旧BCE变化为%+.6f（正值为恶化）。'%(b['refined_abs_error']-a['refined_abs_error'])
        text+='这是固定小型验证集、单种子和局部训练预算下的结果，不能据此声称跨场景重建稳定提升。该起点原本有活跃门控，不能替代早期饱和checkpoint的恢复训练实验。\n'
    else:
        text+='\n截至本次更新，未生成完整训练结果；上段为既定设置和已完成的准备检查，不是已取得的训练收益。\n'
        if interrupted:
            text+='\n按用户要求停止运行。已落盘训练指标步数：%s。已发现配对训练raw深度/置信度轨迹哈希存在差异，原因尚未查明，不作为公平训练对比结论；停止后未继续运行审计或训练。已有输出保留，重新执行探针应使用新的--output_dir。\n'%probe_manifest.get('persisted_train_steps',{})
    text+='''
## 4. 证据文件及结论范围

- `outputs/experiment15_d/run_manifest.json`：每次运行展开后的完整命令、参数、输入checkpoint及活动源码SHA256、退出码和逐配对验收记录。
- `outputs/experiment15_d/synthetic/{cpu,cuda}.json`：实际数值结果；对应目录保存回归和CLI核验记录。
- `outputs/experiment15_d/audit_{relu,leaky_relu}/`：各变体15个case及逐case前向比较。
- `outputs/experiment15_d/audit_summary.json`：只有最终25个配对通过后生成。
- `outputs/experiment15_d/short_train/`：条件满足后生成样本清单、原始逐步指标、各组快照及比较文件。
- 重建本报告：`python summarize_experiment15_d.py`。

本轮改进SGER收益门控的数值学习通路，未改动门控机制、残差范围、margin或融合阈值。点云Acc./Comp./Overall、点数、融合掩码通过率和一致性计数未在本轮采集。项目最终贡献仍须同checkpoint、同raw confidence、同融合协议的raw/refined公平评估，并检查跨checkpoint或重复实验稳定性。
'''
    Path('codex_results/实验15-D门控BCE数值稳定性AB审计结果分析.md').write_text(text)
    from experiment15_d import dump
    dump(ROOT/'analysis_summary.json',{'updated':now,'verified_pairs':len(pairs),'audit_complete':full,
                                     'probe_complete':probe_path.exists(),'groups':aggregated})
    print('Report updated; verified pairs:',len(pairs),'probe:',probe_path.exists())


if __name__=='__main__':main()
