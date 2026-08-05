import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import (
    build_dual_region_geometry,
    compute_normal_from_depth,
    curvature_magnitude,
    depth_gradient_magnitude,
    masked_mean,
    sobel_gradient_magnitude,
)


def _as_depth_map(value):
    if value.dim() == 4:
        assert value.size(1) == 1, "expected a single-channel map"
        return value[:, 0]
    assert value.dim() == 3, "expected shape (B,H,W) or (B,1,H,W)"
    return value


class GeometryCueExtractor(nn.Module):
    def forward(self, depth, normal_pred, confidence, ref_img,
                intrinsics, valid_mask=None):
        depth = _as_depth_map(depth)
        confidence = _as_depth_map(confidence)
        target_size = depth.shape[-2:]

        predicted_valid = torch.isfinite(depth) & (depth > 0)
        if valid_mask is not None:
            predicted_valid = predicted_valid & (_as_depth_map(valid_mask) > 0.5)
        valid_mask = predicted_valid
        depth = torch.where(valid_mask, depth, torch.zeros_like(depth))
        confidence = torch.where(
            torch.isfinite(confidence), confidence,
            torch.zeros_like(confidence))

        if normal_pred.shape[-2:] != target_size:
            normal_pred = F.interpolate(
                normal_pred, size=target_size,
                mode="bilinear", align_corners=False)
            normal_pred = F.normalize(normal_pred, p=2, dim=1, eps=1e-6)
        if ref_img.shape[-2:] != target_size:
            ref_img = F.interpolate(
                ref_img, size=target_size,
                mode="bilinear", align_corners=False)

        depth_mean = masked_mean(depth, valid_mask).detach()
        depth_normalized = depth / (depth_mean + 1e-6)
        normal_depth = compute_normal_from_depth(depth, intrinsics)
        normal_disagreement = 1.0 - torch.abs(
            (normal_pred * normal_depth).sum(dim=1)
        ).clamp(0.0, 1.0)
        image_edge = sobel_gradient_magnitude(ref_img, target_size)
        depth_edge = depth_gradient_magnitude(depth, valid_mask)
        curvature = curvature_magnitude(depth, valid_mask)

        cue_tensor = torch.cat([
            depth_normalized.unsqueeze(1),
            normal_pred,
            normal_depth,
            normal_disagreement.unsqueeze(1),
            confidence.unsqueeze(1),
            image_edge.unsqueeze(1),
            depth_edge.unsqueeze(1),
            curvature.unsqueeze(1),
            ref_img,
        ], dim=1)
        return {
            "cue_tensor": cue_tensor,
            "valid_mask": valid_mask,
            "normal_depth": normal_depth,
            "normal_disagreement": normal_disagreement,
            "image_edge": image_edge,
            "depth_edge": depth_edge,
            "curvature": curvature,
        }


class DualRegionGate(nn.Module):
    def __init__(self, in_channels, hidden_channels=16,
                 threshold_edge=0.25, threshold_depth=0.02,
                 threshold_curv=0.02, conf_mid=0.65,
                 k_conf=10.0, smooth_k=2.0):
        super(DualRegionGate, self).__init__()
        self.threshold_edge = threshold_edge
        self.threshold_depth = threshold_depth
        self.threshold_curv = threshold_curv
        self.conf_mid = conf_mid
        self.k_conf = k_conf
        self.smooth_k = smooth_k
        self.gate_a = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
        )
        self.gate_b = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
        )

    def forward(self, fused_features, ref_img, depth, confidence, valid_mask):
        region_a, weight_b, _ = build_dual_region_geometry(
            ref_img,
            valid_mask,
            depth.detach(),
            confidence.detach(),
            depth.shape[-2:],
            self.threshold_edge,
            self.threshold_depth,
            self.threshold_curv,
            self.conf_mid,
            self.k_conf,
            self.smooth_k,
        )
        region_b = valid_mask & (~region_a)
        learned_a = torch.sigmoid(self.gate_a(fused_features)).squeeze(1)
        learned_b = torch.sigmoid(self.gate_b(fused_features)).squeeze(1)
        gate = region_a.float() * learned_a + weight_b * learned_b
        gate = gate * valid_mask.float()
        return gate.clamp(0.0, 1.0), region_a, region_b, weight_b


class ResidualDepthHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super(ResidualDepthHead, self).__init__()
        reduced_channels = max(hidden_channels // 2, 1)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3,
                      padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, reduced_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Conv2d(reduced_channels, 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features):
        return self.output(self.body(features)).squeeze(1)


class SGERBlock(nn.Module):
    def __init__(self, feature_channels, projected_feature_channels=8,
                 hidden_channels=32, gate_channels=16,
                 max_residual_ratio=2.0, **gate_kwargs):
        super(SGERBlock, self).__init__()
        self.max_residual_ratio = float(max_residual_ratio)
        self.cues = GeometryCueExtractor()
        self.feature_projection = nn.Conv2d(
            feature_channels, projected_feature_channels, 1, bias=False)
        self.uncertainty_channels = 3
        fused_channels = (
            15 + projected_feature_channels + self.uncertainty_channels)
        self.gate = DualRegionGate(
            fused_channels, hidden_channels=gate_channels, **gate_kwargs)
        self.residual = ResidualDepthHead(
            fused_channels, hidden_channels=hidden_channels)

    def forward(self, depth, normal_pred, confidence, ref_img,
                intrinsics, ref_feature, depth_interval, valid_mask=None,
                uncertainty=None):
        depth = _as_depth_map(depth)
        confidence = _as_depth_map(confidence)

        predicted_valid = torch.isfinite(depth) & (depth > 0)
        if valid_mask is not None:
            predicted_valid = predicted_valid & (_as_depth_map(valid_mask) > 0.5)
        depth = torch.where(predicted_valid, depth, torch.zeros_like(depth))
        confidence = torch.where(
            torch.isfinite(confidence), confidence,
            torch.zeros_like(confidence))
        cues = self.cues(
            depth, normal_pred, confidence, ref_img,
            intrinsics, predicted_valid)

        feature = self.feature_projection(ref_feature)
        if feature.shape[-2:] != depth.shape[-2:]:
            feature = F.interpolate(
                feature, size=depth.shape[-2:],
                mode="bilinear", align_corners=False)
        if uncertainty is None:
            uncertainty = depth.new_zeros(
                depth.size(0), self.uncertainty_channels,
                depth.size(1), depth.size(2))
        assert uncertainty.dim() == 4
        assert uncertainty.size(1) == self.uncertainty_channels
        if uncertainty.shape[-2:] != depth.shape[-2:]:
            uncertainty = F.interpolate(
                uncertainty, size=depth.shape[-2:],
                mode="bilinear", align_corners=False)
        uncertainty = torch.where(
            torch.isfinite(uncertainty), uncertainty,
            torch.zeros_like(uncertainty))
        fused = torch.cat([
            cues["cue_tensor"], feature, uncertainty], dim=1)

        benefit_gate, region_a, region_b, weight_b = self.gate(
            fused, ref_img, depth, confidence, cues["valid_mask"])
        residual_logit = self.residual(fused)
        interval = float(depth_interval)
        max_residual = self.max_residual_ratio * interval
        raw_residual = max_residual * torch.tanh(residual_logit)
        gated_residual = benefit_gate * raw_residual
        depth_refined = depth + gated_residual
        residual_ratio = gated_residual.abs() / (interval + 1e-6)

        return {
            "depth_refined": depth_refined,
            "raw_depth_residual": raw_residual,
            "depth_residual": gated_residual,
            "residual_ratio": residual_ratio,
            "benefit_gate": benefit_gate,
            "geometry_gate": benefit_gate,
            "region_a": region_a,
            "region_b": region_b,
            "region_b_weight": weight_b,
            "normal_depth": cues["normal_depth"],
            "normal_disagreement": cues["normal_disagreement"],
        }
