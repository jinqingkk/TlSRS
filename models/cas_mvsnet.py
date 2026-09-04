import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .sger_refinement import SGERBlock

Align_Corners_Range = False


def probability_volume_statistics(prob_volume, depth_values):
    """Return normalized entropy, depth variance, and peak separation."""
    assert prob_volume.dim() == 4, "expected probability shape (B,D,H,W)"
    assert depth_values.shape == prob_volume.shape
    num_depth = prob_volume.size(1)
    if num_depth == 1:
        zeros = prob_volume[:, 0] * 0.0
        return {
            "probability_entropy": zeros,
            "depth_variance": zeros,
            "top1_top2_margin": torch.ones_like(zeros),
        }

    entropy = -(prob_volume * torch.log(prob_volume.clamp_min(1e-12))).sum(1)
    entropy = entropy / math.log(float(num_depth))
    depth_mean = (prob_volume * depth_values).sum(1)
    depth_variance = (
        prob_volume * (depth_values - depth_mean.unsqueeze(1)).pow(2)
    ).sum(1)
    top_probabilities = torch.topk(prob_volume, 2, dim=1)[0]
    top1_top2_margin = top_probabilities[:, 0] - top_probabilities[:, 1]
    return {
        "probability_entropy": entropy,
        "depth_variance": depth_variance,
        "top1_top2_margin": top1_top2_margin,
    }

class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization, prob_volume_init=None):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)
        num_views = len(features)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            #warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
            # warped_volume = homo_warping(src_fea, src_proj[:, 2], ref_proj[:, 2], depth_values)

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        # step 3. cost volume regularization
        cost_reg = cost_regularization(volume_variance)
        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = F.softmax(prob_volume_pre, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)
        uncertainty = probability_volume_statistics(prob_volume, depth_values)

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
            depth_index = depth_index.clamp(min=0, max=num_depth-1)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        return dict(
            {"depth": depth,
             "photometric_confidence": photometric_confidence},
            **uncertainty)


class CascadeMVSNet(nn.Module):
    def __init__(self, refine=False, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1], share_cr=False,
                 grad_method="detach", arch_mode="fpn", cr_base_chs=[8, 8, 8], use_sger=False,
                 use_sger_lite=False,
                 sger_share=False, sger_feature_channels=8, sger_hidden_channels=32,
                 sger_gate_channels=16, sger_max_residual_ratio=0.5,
                 detach_refined_feedback=True, sger_gate_kwargs=None):
        super(CascadeMVSNet, self).__init__()
        if use_sger and use_sger_lite:
            raise ValueError("use_sger and use_sger_lite are mutually exclusive")
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.use_sger = use_sger
        self.use_sger_lite = use_sger_lite
        self.sger_enabled = use_sger or use_sger_lite
        self.sger_share = sger_share
        self.sger_residual_scale = 1.0
        self.detach_refined_feedback = detach_refined_feedback
        self.sger_gate_kwargs = sger_gate_kwargs or {}
        self.num_stage = len(ndepths)
        print("**********netphs:{}, depth_intervals_ratio:{},  grad:{}, chs:{}************".format(ndepths,
              depth_interals_ratio, self.grad_method, self.cr_base_chs))

        assert len(ndepths) == len(depth_interals_ratio)

        self.stage_infos = {
            "stage1":{
                "scale": 4.0,
            },
            "stage2": {
                "scale": 2.0,
            },
            "stage3": {
                "scale": 1.0,
            }
        }

        self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage, arch_mode=self.arch_mode)
        if self.share_cr:
            self.cost_regularization = CostRegNet(in_channels=self.feature.out_channels, base_channels=8)
        else:
            self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=self.feature.out_channels[i],
                                                                 base_channels=self.cr_base_chs[i])
                                                      for i in range(self.num_stage)])
        if self.refine:
            self.refine_network = RefineNet()
        self.DepthNet = DepthNet()
        self.normal_head = nn.ModuleList([
            NormalHead(out_channels + 2) for out_channels in self.feature.out_channels[:self.num_stage]
        ])
        if self.sger_enabled:
            if self.sger_share:
                self.sger_feature_adapters = nn.ModuleList([
                    nn.Conv2d(out_channels, sger_feature_channels, 1, bias=False)
                    for out_channels in self.feature.out_channels[:self.num_stage]
                ])
                self.shared_sger = SGERBlock(
                    feature_channels=sger_feature_channels,
                    projected_feature_channels=sger_feature_channels,
                    hidden_channels=sger_hidden_channels,
                    gate_channels=sger_gate_channels,
                    max_residual_ratio=sger_max_residual_ratio,
                    **self.sger_gate_kwargs)
            else:
                self.sger_blocks = nn.ModuleList([
                    SGERBlock(
                        feature_channels=out_channels,
                        projected_feature_channels=sger_feature_channels,
                        hidden_channels=sger_hidden_channels,
                        gate_channels=sger_gate_channels,
                        max_residual_ratio=sger_max_residual_ratio,
                        **self.sger_gate_kwargs)
                    for out_channels in (
                        [self.feature.out_channels[self.num_stage - 1]]
                        if self.use_sger_lite
                        else self.feature.out_channels[:self.num_stage])
                ])

    def set_sger_residual_scale(self, scale):
        scale = float(scale)
        if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
            raise ValueError(
                "sger residual scale must be finite and in [0, 1]")
        self.sger_residual_scale = scale

    def forward(self, imgs, proj_matrices, depth_values):
        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / depth_values.size(1)

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):  #imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))

        outputs = {}
        depth, cur_depth = None, None
        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            #stage feature, proj_mats, scales
            stage_key = "stage{}".format(stage_idx + 1)
            features_stage = [feat[stage_key] for feat in features]
            proj_matrices_stage = proj_matrices[stage_key]
            stage_scale = self.stage_infos[stage_key]["scale"]

            if depth is not None:
                if self.use_sger and self.detach_refined_feedback:
                    cur_depth = depth.detach()
                elif (not self.use_sger) and self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                                                [img.shape[2], img.shape[3]], mode='bilinear',
                                                align_corners=Align_Corners_Range).squeeze(1)
            else:
                cur_depth = depth_values
            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                                                        ndepth=self.ndepths[stage_idx],
                                                        depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                                                        dtype=img[0].dtype,
                                                        device=img[0].device,
                                                        shape=[img.shape[0], img.shape[2], img.shape[3]],
                                                        max_depth=depth_max,
                                                        min_depth=depth_min)

            outputs_stage = self.DepthNet(features_stage, proj_matrices_stage,
                                          depth_values=F.interpolate(depth_range_samples.unsqueeze(1),
                                                                     [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)], mode='trilinear',
                                                                     align_corners=Align_Corners_Range).squeeze(1),
                                          num_depth=self.ndepths[stage_idx],
                                          cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx])

            depth_raw = outputs_stage["depth"]
            ref_feature = features_stage[0]
            run_sger = self.use_sger or (
                self.use_sger_lite and stage_idx == self.num_stage - 1)
            isolate_sger_lite = (
                self.use_sger_lite and stage_idx == self.num_stage - 1)
            run_normal_head = (not self.use_sger_lite) or (stage_idx == self.num_stage - 1)
            if run_normal_head:
                depth_input = depth_raw.unsqueeze(1)
                confidence_input = outputs_stage["photometric_confidence"].unsqueeze(1)
                normal_feature = ref_feature
                if isolate_sger_lite:
                    normal_feature = normal_feature.detach()
                    depth_input = depth_input.detach()
                    confidence_input = confidence_input.detach()
                if depth_input.shape[-2:] != ref_feature.shape[-2:]:
                    depth_input = F.interpolate(depth_input, size=ref_feature.shape[-2:],
                                                mode='bilinear',
                                                align_corners=Align_Corners_Range)
                if confidence_input.shape[-2:] != ref_feature.shape[-2:]:
                    confidence_input = F.interpolate(confidence_input, size=ref_feature.shape[-2:],
                                                     mode='bilinear',
                                                     align_corners=Align_Corners_Range)
                normal_input = torch.cat([
                    normal_feature, depth_input, confidence_input], dim=1)
                outputs_stage["normal"] = self.normal_head[stage_idx](normal_input)

            if run_sger:
                ref_img_stage = F.interpolate(
                    imgs[:, 0], size=depth_raw.shape[-2:],
                    mode='bilinear', align_corners=Align_Corners_Range)
                intrinsics_stage = proj_matrices_stage[:, 0, 1, :3, :3]
                sger_backbone_feature = (
                    ref_feature.detach()
                    if isolate_sger_lite else ref_feature)
                if self.sger_share:
                    sger_feature = self.sger_feature_adapters[stage_idx](
                        sger_backbone_feature)
                    sger_block = self.shared_sger
                else:
                    sger_feature = sger_backbone_feature
                    sger_block = self.sger_blocks[0 if self.use_sger_lite else stage_idx]
                stage_interval = (
                    self.depth_interals_ratio[stage_idx] * depth_interval)
                uncertainty = torch.stack([
                    outputs_stage["probability_entropy"],
                    outputs_stage["depth_variance"] / (
                        stage_interval * stage_interval + 1e-6),
                    outputs_stage["top1_top2_margin"],
                ], dim=1)
                if isolate_sger_lite:
                    uncertainty = uncertainty.detach()
                sger_outputs = sger_block(
                    depth_raw.detach(),
                    outputs_stage["normal"],
                    outputs_stage["photometric_confidence"].detach()
                    if isolate_sger_lite
                    else outputs_stage["photometric_confidence"],
                    ref_img_stage,
                    intrinsics_stage,
                    sger_feature,
                    depth_interval=stage_interval,
                    uncertainty=uncertainty)
                effective_residual = (
                    self.sger_residual_scale
                    * sger_outputs["depth_residual"])
                outputs_stage["depth_raw"] = depth_raw
                outputs_stage["depth_interval"] = depth_raw.new_tensor(
                    stage_interval)
                outputs_stage["raw_depth_residual"] = (
                    sger_outputs["raw_depth_residual"])
                outputs_stage["depth_residual"] = effective_residual
                outputs_stage["residual_ratio"] = (
                    self.sger_residual_scale
                    * sger_outputs["residual_ratio"])
                refined_depth_base = (
                    depth_raw.detach() if isolate_sger_lite else depth_raw)
                outputs_stage["depth_refined"] = (
                    refined_depth_base + effective_residual)
                outputs_stage["depth"] = outputs_stage["depth_refined"]
                outputs_stage["sger_residual_scale"] = depth_raw.new_tensor(
                    self.sger_residual_scale)
                for key in ("benefit_gate", "geometry_gate", "region_a",
                            "region_b_weight", "normal_disagreement"):
                    outputs_stage[key] = sger_outputs[key]
                depth = outputs_stage["depth_refined"]
            else:
                depth = depth_raw

            outputs[stage_key] = outputs_stage
            outputs.update(outputs_stage)

        # depth map refinement
        if self.refine:
            refined_depth = self.refine_network(torch.cat((imgs[:, 0], depth), 1))
            outputs["refined_depth"] = refined_depth

        return outputs
