import ast
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import models.cas_mvsnet as cas_module
from models.cas_mvsnet import CascadeMVSNet, probability_volume_statistics
from models.module import (
    bounded_residual_target,
    cas_mvsnet_loss,
    residual_benefit_target,
)
from models.sger_refinement import SGERBlock


def make_sger_inputs(batch=2, height=8, width=12, feature_channels=8):
    depth = torch.ones(batch, height, width)
    normal = torch.zeros(batch, 3, height, width)
    normal[:, 2] = 1.0
    confidence = torch.full((batch, height, width), 0.9)
    ref_img = torch.zeros(batch, 3, height, width)
    intrinsics = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
    ref_feature = torch.randn(batch, feature_channels, height, width)
    return depth, normal, confidence, ref_img, intrinsics, ref_feature


def test_probability_volume_statistics_measure_uncertainty():
    probability = torch.full((1, 4, 2, 3), 0.25)
    depth_values = torch.arange(1.0, 5.0).view(1, 4, 1, 1).expand_as(
        probability)

    statistics = probability_volume_statistics(probability, depth_values)

    assert set(statistics) == {
        "probability_entropy", "depth_variance", "top1_top2_margin"}
    for value in statistics.values():
        assert value.shape == (1, 2, 3)
        assert torch.isfinite(value).all()
    assert torch.allclose(
        statistics["probability_entropy"], torch.ones(1, 2, 3))
    assert torch.allclose(
        statistics["depth_variance"], torch.full((1, 2, 3), 1.25))
    assert torch.allclose(
        statistics["top1_top2_margin"], torch.zeros(1, 2, 3))


def test_probability_volume_statistics_handle_single_depth_bin():
    probability = torch.ones(1, 1, 2, 3)
    depth_values = torch.full_like(probability, 2.0)

    statistics = probability_volume_statistics(probability, depth_values)

    assert torch.all(statistics["probability_entropy"] == 0)
    assert torch.all(statistics["depth_variance"] == 0)
    assert torch.all(statistics["top1_top2_margin"] == 1)


def test_sger_block_returns_bounded_outputs():
    block = SGERBlock(feature_channels=8, hidden_channels=16,
                      max_residual_ratio=2.0)
    inputs = make_sger_inputs()

    outputs = block(*inputs, depth_interval=0.5)

    assert outputs["depth_refined"].shape == (2, 8, 12)
    assert outputs["depth_residual"].shape == (2, 8, 12)
    assert outputs["geometry_gate"].shape == (2, 8, 12)
    assert outputs["region_a"].dtype == torch.bool
    assert torch.isfinite(outputs["depth_refined"]).all()
    assert outputs["geometry_gate"].min().item() >= 0.0
    assert outputs["geometry_gate"].max().item() <= 1.0
    assert outputs["depth_residual"].abs().max().item() <= 1.0 + 1e-6
    assert torch.allclose(outputs["depth_refined"], inputs[0], atol=1e-7)


def test_sger_block_separates_residual_proposal_and_benefit_gate():
    block = SGERBlock(feature_channels=8, hidden_channels=16,
                      max_residual_ratio=0.25)
    inputs = make_sger_inputs()
    uncertainty = torch.stack([
        torch.full_like(inputs[0], 0.8),
        torch.full_like(inputs[0], 0.1),
        torch.full_like(inputs[0], 0.2),
    ], dim=1)
    with torch.no_grad():
        block.residual.output.bias.fill_(0.2)

    outputs = block(
        *inputs, depth_interval=0.5, uncertainty=uncertainty)

    assert outputs["raw_depth_residual"].shape == inputs[0].shape
    assert outputs["benefit_gate"].shape == inputs[0].shape
    assert torch.allclose(
        outputs["depth_residual"],
        outputs["benefit_gate"] * outputs["raw_depth_residual"])
    assert torch.allclose(outputs["geometry_gate"], outputs["benefit_gate"])
    assert outputs["raw_depth_residual"].abs().max().item() <= 0.125 + 1e-6


def test_sger_regions_ignore_invalid_depth_and_are_disjoint():
    block = SGERBlock(feature_channels=8, hidden_channels=16)
    depth, normal, confidence, ref_img, intrinsics, feature = make_sger_inputs()
    depth[:, 0, 0] = 0.0

    outputs = block(depth, normal, confidence, ref_img, intrinsics,
                    feature, depth_interval=0.5)

    region_b = outputs["region_b_weight"] > 0
    assert not (outputs["region_a"] & region_b).any()
    assert not outputs["region_a"][:, 0, 0].any()
    assert torch.all(outputs["region_b_weight"][:, 0, 0] == 0)
    assert torch.all(outputs["geometry_gate"][:, 0, 0] == 0)


def test_sger_sanitizes_nonfinite_depth_before_geometry_processing():
    block = SGERBlock(feature_channels=8, hidden_channels=16)
    depth, normal, confidence, ref_img, intrinsics, feature = make_sger_inputs()
    depth[:, 0, 0] = float("nan")

    outputs = block(depth, normal, confidence, ref_img, intrinsics,
                    feature, depth_interval=0.5)

    assert torch.isfinite(outputs["depth_refined"]).all()
    assert torch.all(outputs["depth_refined"][:, 0, 0] == 0)
    assert torch.all(outputs["geometry_gate"][:, 0, 0] == 0)


def test_sger_refined_depth_backpropagates_through_learned_path():
    block = SGERBlock(feature_channels=8, hidden_channels=16)
    depth, normal, confidence, ref_img, intrinsics, feature = make_sger_inputs()
    depth.requires_grad_()
    normal.requires_grad_()
    feature.requires_grad_()
    torch.nn.init.constant_(block.residual.output.weight, 0.01)

    outputs = block(depth, normal, confidence, ref_img, intrinsics,
                    feature, depth_interval=0.5)
    outputs["depth_refined"].mean().backward()

    assert depth.grad is not None
    assert normal.grad is not None
    assert feature.grad is not None
    assert block.residual.output.weight.grad is not None
    assert not outputs["region_a"].requires_grad
    assert not outputs["region_b_weight"].requires_grad


def _identity_proj(batch, views):
    return torch.eye(4).view(1, 1, 1, 4, 4).repeat(
        batch, views, 2, 1, 1)


def test_cascade_sger_returns_raw_and_refined_depth_for_every_stage():
    torch.manual_seed(4)
    model = CascadeMVSNet(
        ndepths=[8, 8, 8],
        depth_interals_ratio=[4, 2, 1],
        use_sger=True,
        sger_hidden_channels=16,
        sger_gate_channels=8,
    )
    model.eval()
    imgs = torch.rand(1, 3, 3, 32, 32)
    depth_values = torch.linspace(1.0, 2.0, 8).view(1, 8)
    proj_matrices = {
        "stage1": _identity_proj(1, 3),
        "stage2": _identity_proj(1, 3),
        "stage3": _identity_proj(1, 3),
    }

    with torch.no_grad():
        outputs = model(imgs, proj_matrices, depth_values)

    for stage_key in ("stage1", "stage2", "stage3"):
        stage = outputs[stage_key]
        assert "depth_raw" in stage
        assert "depth_refined" in stage
        assert "depth_residual" in stage
        assert "geometry_gate" in stage
        assert torch.allclose(stage["depth"], stage["depth_refined"])
        assert torch.allclose(stage["depth_raw"], stage["depth_refined"], atol=1e-7)
    assert torch.allclose(outputs["depth"], outputs["stage3"]["depth_refined"])


def test_cascade_sger_lite_refines_stage3_only():
    torch.manual_seed(5)
    model = CascadeMVSNet(
        ndepths=[8, 8, 8],
        depth_interals_ratio=[4, 2, 1],
        use_sger_lite=True,
        sger_hidden_channels=16,
        sger_gate_channels=8,
        sger_max_residual_ratio=0.5,
    )
    model.eval()
    imgs = torch.rand(1, 3, 3, 32, 32)
    depth_values = torch.linspace(1.0, 2.0, 8).view(1, 8)
    proj_matrices = {
        "stage1": _identity_proj(1, 3),
        "stage2": _identity_proj(1, 3),
        "stage3": _identity_proj(1, 3),
    }

    with torch.no_grad():
        outputs = model(imgs, proj_matrices, depth_values)

    for stage_key in ("stage1", "stage2"):
        assert "normal" not in outputs[stage_key]
        assert "depth_raw" not in outputs[stage_key]
        assert "depth_refined" not in outputs[stage_key]
        assert "depth_residual" not in outputs[stage_key]
        assert "geometry_gate" not in outputs[stage_key]
        assert torch.allclose(outputs[stage_key]["depth"], outputs[stage_key]["depth"])
    stage3 = outputs["stage3"]
    assert "normal" in stage3
    assert "depth_raw" in stage3
    assert "depth_refined" in stage3
    assert "depth_residual" in stage3
    assert "geometry_gate" in stage3
    assert "benefit_gate" in stage3
    assert "raw_depth_residual" in stage3
    assert "probability_entropy" in stage3
    assert "depth_variance" in stage3
    assert "top1_top2_margin" in stage3
    assert torch.allclose(stage3["depth"], stage3["depth_refined"])
    assert torch.allclose(outputs["depth"], stage3["depth_refined"])
    assert torch.allclose(outputs["depth_raw"], stage3["depth_raw"])
    assert torch.allclose(outputs["depth_refined"], stage3["depth_refined"])


def test_cascade_sger_lite_scales_effective_residual():
    torch.manual_seed(6)
    model = CascadeMVSNet(
        ndepths=[8, 8, 8],
        depth_interals_ratio=[4, 2, 1],
        use_sger_lite=True,
        sger_hidden_channels=16,
        sger_gate_channels=8,
    )
    model.eval()
    with torch.no_grad():
        model.sger_blocks[0].residual.output.bias.fill_(0.2)
    imgs = torch.rand(1, 3, 3, 32, 32)
    depth_values = torch.linspace(1.0, 2.0, 8).view(1, 8)
    proj_matrices = {
        key: _identity_proj(1, 3)
        for key in ("stage1", "stage2", "stage3")
    }

    with torch.no_grad():
        model.set_sger_residual_scale(0.0)
        scale_zero = model(imgs, proj_matrices, depth_values)
        model.set_sger_residual_scale(0.5)
        scale_half = model(imgs, proj_matrices, depth_values)
        model.set_sger_residual_scale(1.0)
        scale_full = model(imgs, proj_matrices, depth_values)

    assert torch.allclose(
        scale_zero["depth_refined"], scale_zero["depth_raw"])
    assert torch.all(scale_zero["depth_residual"] == 0)
    assert torch.allclose(
        scale_half["depth_residual"],
        0.5 * scale_full["depth_residual"],
        atol=1e-7)
    assert torch.allclose(
        scale_half["depth_refined"],
        scale_half["depth_raw"] + scale_half["depth_residual"])
    assert scale_half["sger_residual_scale"].item() == 0.5
    assert "sger_residual_scale" not in model.state_dict()


def test_cascade_sger_lite_refined_path_isolates_backbone_gradients():
    torch.manual_seed(7)
    model = CascadeMVSNet(
        ndepths=[8, 8, 8],
        depth_interals_ratio=[4, 2, 1],
        use_sger_lite=True,
        sger_hidden_channels=16,
        sger_gate_channels=8,
    )
    model.train()
    with torch.no_grad():
        model.sger_blocks[0].residual.output.weight.fill_(0.01)
        model.sger_blocks[0].residual.output.bias.fill_(0.2)
    imgs = torch.rand(2, 3, 3, 32, 32)
    depth_values = torch.linspace(1.0, 2.0, 8).view(1, 8).repeat(2, 1)
    proj_matrices = {
        key: _identity_proj(2, 3)
        for key in ("stage1", "stage2", "stage3")
    }

    outputs = model(imgs, proj_matrices, depth_values)
    outputs["stage3"]["depth_refined"].mean().backward()

    backbone_parameters = list(model.feature.parameters())
    backbone_parameters += list(model.cost_regularization.parameters())
    assert all(parameter.grad is None for parameter in backbone_parameters)
    assert any(
        parameter.grad is not None
        and parameter.grad.abs().sum().item() > 0
        for parameter in model.normal_head[2].parameters())
    assert any(
        parameter.grad is not None
        and parameter.grad.abs().sum().item() > 0
        for parameter in model.sger_blocks[0].parameters())


def test_cascade_sger_residual_scale_rejects_invalid_values():
    model = CascadeMVSNet(
        ndepths=[8, 8, 8],
        depth_interals_ratio=[4, 2, 1],
        use_sger_lite=True,
    )
    assert model.sger_residual_scale == 1.0

    for scale in (-0.1, 1.1, float("nan"), float("inf")):
        try:
            model.set_sger_residual_scale(scale)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid residual scale must raise ValueError")


def test_cascade_disabled_path_has_no_sger_only_outputs():
    model = CascadeMVSNet(
        ndepths=[8, 8, 8],
        depth_interals_ratio=[4, 2, 1],
        use_sger=False,
    )
    model.eval()
    imgs = torch.rand(1, 3, 3, 32, 32)
    depth_values = torch.linspace(1.0, 2.0, 8).view(1, 8)
    proj_matrices = {
        "stage1": _identity_proj(1, 3),
        "stage2": _identity_proj(1, 3),
        "stage3": _identity_proj(1, 3),
    }

    with torch.no_grad():
        outputs = model(imgs, proj_matrices, depth_values)

    for stage_key in ("stage1", "stage2", "stage3"):
        assert "depth_raw" not in outputs[stage_key]
        assert "depth_refined" not in outputs[stage_key]
        assert "geometry_gate" not in outputs[stage_key]


def test_cascade_shared_sger_mode_runs_all_stages():
    model = CascadeMVSNet(
        ndepths=[8, 8, 8],
        depth_interals_ratio=[4, 2, 1],
        use_sger=True,
        sger_share=True,
        sger_hidden_channels=16,
        sger_gate_channels=8,
    )
    model.eval()
    imgs = torch.rand(1, 3, 3, 32, 32)
    depth_values = torch.linspace(1.0, 2.0, 8).view(1, 8)
    proj_matrices = {
        "stage1": _identity_proj(1, 3),
        "stage2": _identity_proj(1, 3),
        "stage3": _identity_proj(1, 3),
    }

    with torch.no_grad():
        outputs = model(imgs, proj_matrices, depth_values)

    assert hasattr(model, "shared_sger")
    assert len(model.sger_feature_adapters) == 3
    for stage_key in ("stage1", "stage2", "stage3"):
        assert "depth_refined" in outputs[stage_key]


def test_cascade_uses_refined_depth_as_next_stage_sampling_center():
    model = CascadeMVSNet(
        ndepths=[8, 8, 8],
        depth_interals_ratio=[4, 2, 1],
        use_sger=True,
        sger_hidden_channels=16,
        sger_gate_channels=8,
    )
    model.eval()
    with torch.no_grad():
        model.sger_blocks[0].residual.output.bias.fill_(0.2)
    imgs = torch.rand(1, 3, 3, 32, 32)
    depth_values = torch.linspace(1.0, 2.0, 8).view(1, 8)
    proj_matrices = {
        "stage1": _identity_proj(1, 3),
        "stage2": _identity_proj(1, 3),
        "stage3": _identity_proj(1, 3),
    }
    captured_centers = []
    original_sampler = cas_module.get_depth_range_samples

    def capture_sampler(*args, **kwargs):
        captured_centers.append(kwargs["cur_depth"].detach().clone())
        return original_sampler(*args, **kwargs)

    cas_module.get_depth_range_samples = capture_sampler
    try:
        with torch.no_grad():
            outputs = model(imgs, proj_matrices, depth_values)
    finally:
        cas_module.get_depth_range_samples = original_sampler

    expected_center = F.interpolate(
        outputs["stage1"]["depth_refined"].unsqueeze(1),
        size=(32, 32), mode="bilinear", align_corners=False).squeeze(1)
    assert not torch.allclose(
        outputs["stage1"]["depth_raw"],
        outputs["stage1"]["depth_refined"])
    assert torch.allclose(captured_centers[1], expected_center)


def _unpack_loss(result):
    return result[0], result[1], result[-1]


def test_sger_loss_supervises_raw_and_refined_depth_separately():
    raw = torch.full((1, 2, 3), 2.0, requires_grad=True)
    refined = torch.ones(1, 2, 3, requires_grad=True)
    outputs = {"stage1": {
        "depth": refined,
        "depth_raw": raw,
        "depth_residual": torch.ones(1, 2, 3),
        "geometry_gate": torch.full((1, 2, 3), 0.5),
        "region_a": torch.tensor([[[True, False, False],
                                    [True, False, False]]]),
        "region_b_weight": torch.full((1, 2, 3), 0.25),
    }}
    depth_gt = {"stage1": torch.zeros(1, 2, 3)}
    mask = {"stage1": torch.ones(1, 2, 3)}

    total, depth_loss, extra = _unpack_loss(cas_mvsnet_loss(
        outputs,
        depth_gt,
        mask,
        dlossw=[1.0],
        raw_depth_loss_weight=0.5,
        refined_depth_loss_weight=1.0,
        return_extra=True,
    ))

    assert torch.allclose(extra["stage1/raw_depth_loss"], torch.tensor(1.5))
    assert torch.allclose(extra["stage1/refined_depth_loss"], torch.tensor(0.5))
    assert torch.allclose(extra["stage1/mean_abs_depth_residual"], torch.tensor(1.0))
    assert torch.allclose(extra["stage1/geometry_gate_mean"], torch.tensor(0.5))
    assert torch.allclose(extra["stage1/region_A_ratio"], torch.tensor(1.0 / 3.0))
    assert torch.allclose(extra["stage1/region_B_weight_mean"], torch.tensor(0.25))
    assert torch.allclose(extra["stage1/raw_to_refined_error_delta"], torch.tensor(1.0))
    assert torch.allclose(depth_loss, torch.tensor(0.5))
    assert torch.allclose(total, torch.tensor(1.25))


def test_bounded_residual_target_has_correct_direction_and_limit():
    depth_raw = torch.tensor([[[2.0, 0.0]]], requires_grad=True)
    depth_gt = torch.tensor([[[3.0, -3.0]]])

    target = bounded_residual_target(
        depth_raw, depth_gt, depth_interval=2.0,
        max_residual_ratio=0.25)

    assert torch.allclose(target, torch.tensor([[[0.5, -0.5]]]))
    assert not target.requires_grad


def test_residual_benefit_target_requires_margin_improvement():
    depth_raw = torch.tensor([[[2.0, 2.0]]])
    raw_residual = torch.tensor([[[-0.8, 0.8]]], requires_grad=True)
    depth_gt = torch.ones(1, 1, 2)

    target = residual_benefit_target(
        depth_raw, raw_residual, depth_gt,
        depth_interval=1.0, margin_ratio=0.05)

    assert torch.equal(target, torch.tensor([[[True, False]]]))
    assert not target.requires_grad


def test_sger_benefit_supervision_reports_selective_refinement_metrics():
    raw = torch.tensor([[[2.0, 2.0]]], requires_grad=True)
    proposal = torch.tensor([[[-0.8, 0.8]]], requires_grad=True)
    gate = torch.tensor([[[0.8, 0.2]]], requires_grad=True)
    refined = raw.detach() + gate * proposal
    outputs = {"stage1": {
        "depth": refined,
        "depth_raw": raw,
        "raw_depth_residual": proposal,
        "depth_residual": gate * proposal,
        "benefit_gate": gate,
        "geometry_gate": gate,
        "depth_interval": raw.new_tensor(1.0),
    }}
    depth_gt = {"stage1": torch.ones(1, 1, 2)}
    mask = {"stage1": torch.ones(1, 1, 2)}

    total, _, extra = _unpack_loss(cas_mvsnet_loss(
        outputs, depth_gt, mask,
        dlossw=[0.0],
        raw_depth_loss_weight=0.0,
        refined_depth_loss_weight=0.0,
        residual_target_loss_weight=1.0,
        gate_benefit_loss_weight=1.0,
        residual_target_ratio=0.25,
        benefit_margin_ratio=0.05,
        return_extra=True))

    assert total.item() > 0
    assert torch.allclose(
        extra["stage1/raw_abs_error"], torch.tensor(1.0))
    assert torch.allclose(
        extra["stage1/refined_abs_error"], torch.tensor(0.76))
    assert torch.allclose(
        extra["stage1/benefit_gate_mean"], torch.tensor(0.5))
    assert torch.allclose(
        extra["stage1/refined_improved_pixel_ratio"], torch.tensor(0.5))
    assert torch.allclose(
        extra["stage1/refined_worsened_pixel_ratio"], torch.tensor(0.5))
    assert torch.allclose(
        extra["stage1/gate_on_improved_mean"], torch.tensor(0.8))
    assert torch.allclose(
        extra["stage1/gate_on_worsened_mean"], torch.tensor(0.2))
    assert torch.allclose(
        extra["stage1/residual_sign_accuracy"], torch.tensor(0.5))
    assert torch.allclose(
        extra["stage1/benefit_target_positive_ratio"], torch.tensor(0.5))
    assert "stage1/residual_target_loss" in extra
    assert "stage1/gate_benefit_loss" in extra
    total.backward()
    assert proposal.grad is not None
    assert gate.grad is not None
    assert raw.grad is not None
    assert torch.all(raw.grad == 0)


def test_sger_loss_regularizes_normalized_gated_residual():
    depth = torch.zeros(1, 2, 3, requires_grad=True)
    outputs = {"stage1": {
        "depth": depth,
        "depth_raw": depth,
        "residual_ratio": torch.full((1, 2, 3), 2.0),
    }}
    depth_gt = {"stage1": torch.zeros(1, 2, 3)}
    mask = {"stage1": torch.ones(1, 2, 3)}

    total, _, extra = _unpack_loss(cas_mvsnet_loss(
        outputs,
        depth_gt,
        mask,
        dlossw=[1.0],
        raw_depth_loss_weight=0.0,
        refined_depth_loss_weight=1.0,
        residual_loss_weight=0.1,
        return_extra=True,
    ))

    assert torch.allclose(extra["stage1/residual_loss"], torch.tensor(2.0))
    assert torch.allclose(total, torch.tensor(0.2))


def test_sger_gate_loss_uses_only_valid_pixels():
    depth = torch.zeros(1, 2, 2, requires_grad=True)
    outputs = {"stage1": {
        "depth": depth,
        "geometry_gate": torch.tensor([[[0.2, 0.4], [0.8, 1.0]]]),
    }}
    depth_gt = {"stage1": torch.zeros(1, 2, 2)}
    mask = {"stage1": torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])}

    total, _, extra = _unpack_loss(cas_mvsnet_loss(
        outputs,
        depth_gt,
        mask,
        dlossw=[1.0],
        gate_loss_weight=0.5,
        return_extra=True,
    ))

    assert torch.allclose(extra["stage1/gate_loss"], torch.tensor(0.3))
    assert torch.allclose(total, torch.tensor(0.15))


def test_safe_refine_loss_penalizes_only_regressions():
    raw = torch.tensor([[[2.0, 2.0]]], requires_grad=True)
    refined = torch.tensor([[[1.0, 3.0]]], requires_grad=True)
    outputs = {"stage1": {"depth": refined, "depth_raw": raw}}
    depth_gt = {"stage1": torch.ones(1, 1, 2)}
    mask = {"stage1": torch.ones(1, 1, 2)}

    total, _, extra = _unpack_loss(cas_mvsnet_loss(
        outputs,
        depth_gt,
        mask,
        dlossw=[0.0],
        raw_depth_loss_weight=0.0,
        refined_depth_loss_weight=0.0,
        safe_refine_loss_weight=0.1,
        safe_refine_margin=0.0,
        return_extra=True,
    ))

    assert torch.allclose(
        extra["stage1/safe_refine_loss"], torch.tensor(0.5))
    assert torch.allclose(total, torch.tensor(0.05))
    total.backward()
    assert raw.grad is not None
    assert torch.allclose(raw.grad, torch.zeros_like(raw.grad))
    assert refined.grad is not None


def test_safe_refine_loss_is_zero_for_improvement_and_applies_margin():
    raw = torch.tensor([[[2.0, 2.0]]])
    refined = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
    outputs = {"stage1": {"depth": refined, "depth_raw": raw}}
    depth_gt = {"stage1": torch.ones(1, 1, 2)}
    mask = {"stage1": torch.ones(1, 1, 2)}

    total_zero, _, extra_zero = _unpack_loss(cas_mvsnet_loss(
        outputs,
        depth_gt,
        mask,
        dlossw=[0.0],
        raw_depth_loss_weight=0.0,
        refined_depth_loss_weight=0.0,
        safe_refine_loss_weight=0.1,
        safe_refine_margin=0.0,
        return_extra=True,
    ))
    total_margin, _, extra_margin = _unpack_loss(cas_mvsnet_loss(
        outputs,
        depth_gt,
        mask,
        dlossw=[0.0],
        raw_depth_loss_weight=0.0,
        refined_depth_loss_weight=0.0,
        safe_refine_loss_weight=0.1,
        safe_refine_margin=0.2,
        return_extra=True,
    ))

    assert torch.allclose(
        extra_zero["stage1/safe_refine_loss"], torch.tensor(0.0))
    assert torch.allclose(total_zero, torch.tensor(0.0))
    assert torch.allclose(
        extra_margin["stage1/safe_refine_loss"], torch.tensor(0.1))
    assert torch.allclose(total_margin, torch.tensor(0.01))


def test_sger_loss_scale_warms_all_refinement_terms():
    raw = torch.full((1, 1, 2), 2.0, requires_grad=True)
    refined = torch.full((1, 1, 2), 3.0, requires_grad=True)
    outputs = {"stage1": {
        "depth": refined,
        "depth_raw": raw,
        "residual_ratio": torch.full((1, 1, 2), 2.0),
        "geometry_gate": torch.full((1, 1, 2), 0.5),
    }}
    depth_gt = {"stage1": torch.zeros(1, 1, 2)}
    mask = {"stage1": torch.ones(1, 1, 2)}

    totals = []
    extras = []
    for scale in (0.0, 0.5, 1.0):
        total, _, extra = _unpack_loss(cas_mvsnet_loss(
            outputs,
            depth_gt,
            mask,
            dlossw=[1.0],
            raw_depth_loss_weight=1.0,
            refined_depth_loss_weight=1.0,
            residual_loss_weight=0.1,
            gate_loss_weight=0.2,
            safe_refine_loss_weight=0.3,
            sger_loss_scale=scale,
            return_extra=True,
        ))
        totals.append(total)
        extras.append(extra)

    assert torch.allclose(totals[0], torch.tensor(1.5))
    assert torch.allclose(totals[1], torch.tensor(3.05))
    assert torch.allclose(totals[2], torch.tensor(4.6))
    expected_metrics = {
        "sger_residual_scale": 0.5,
        "effective_refined_depth_loss_weight": 0.5,
        "effective_residual_loss_weight": 0.05,
        "effective_gate_loss_weight": 0.1,
        "effective_safe_refine_loss_weight": 0.15,
    }
    for key, expected in expected_metrics.items():
        assert torch.allclose(extras[1][key], torch.tensor(expected))


def test_sger_loss_scale_logs_refined_improved_pixel_ratio():
    raw = torch.tensor([[[2.0, 2.0]]])
    refined = torch.tensor([[[1.0, 3.0]]], requires_grad=True)
    outputs = {"stage1": {"depth": refined, "depth_raw": raw}}
    depth_gt = {"stage1": torch.ones(1, 1, 2)}
    mask = {"stage1": torch.ones(1, 1, 2)}

    _, _, extra = _unpack_loss(cas_mvsnet_loss(
        outputs, depth_gt, mask, dlossw=[1.0], return_extra=True))

    assert torch.allclose(
        extra["stage1/refined_improved_pixel_ratio"], torch.tensor(0.5))


def test_sger_loss_scale_rejects_invalid_values():
    depth = torch.zeros(1, 1, 1, requires_grad=True)
    outputs = {"stage1": {"depth": depth}}
    depth_gt = {"stage1": torch.zeros(1, 1, 1)}
    mask = {"stage1": torch.ones(1, 1, 1)}

    for scale in (-0.1, 1.1, float("nan"), float("inf")):
        try:
            cas_mvsnet_loss(
                outputs, depth_gt, mask,
                sger_loss_scale=scale, return_extra=True)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid SGER loss scale must raise ValueError")


def _load_train_helpers(*names):
    train_path = Path(__file__).resolve().parents[1] / "train.py"
    tree = ast.parse(train_path.read_text())
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {}
    exec(compile(ast.Module(body=selected), str(train_path), "exec"), namespace)
    return [namespace[name] for name in names]


def test_sger_lite_optimizer_groups_cover_parameters_once():
    _, build_optimizer_param_groups = _load_train_helpers(
        "is_sger_lite_parameter", "build_optimizer_param_groups")
    model = CascadeMVSNet(
        ndepths=[8, 8, 8],
        depth_interals_ratio=[4, 2, 1],
        use_sger_lite=True,
    )

    groups = build_optimizer_param_groups(
        model, base_lr=0.001, backbone_lr_scale=0.1)

    assert [group["lr"] for group in groups] == [0.001, 0.0001]
    grouped = [id(param) for group in groups for param in group["params"]]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(param) for param in model.parameters()}

    names_by_id = {id(param): name for name, param in model.named_parameters()}
    sger_names = {names_by_id[id(param)] for param in groups[0]["params"]}
    backbone_names = {names_by_id[id(param)] for param in groups[1]["params"]}
    assert any(name.startswith("normal_head.2.") for name in sger_names)
    assert any(name.startswith("sger_blocks.") for name in sger_names)
    assert any(name.startswith("feature.") for name in backbone_names)


def test_sger_warmup_schedule_matches_experiment13_epochs():
    compute_sger_warmup_scale, = _load_train_helpers(
        "compute_sger_warmup_scale")

    actual = [
        compute_sger_warmup_scale(epoch, 3, 6)
        for epoch in range(8)
    ]

    assert actual == [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
    assert compute_sger_warmup_scale(2, 3, 3) == 0.0
    assert compute_sger_warmup_scale(3, 3, 3) == 1.0


def test_experiment14_loss_schedule_separates_proposal_and_gate_training():
    compute_scales, = _load_train_helpers(
        "compute_experiment14_loss_scales")

    assert compute_scales(2) == (0.0, 0.0)
    assert compute_scales(3) == (0.0, 0.0)
    assert compute_scales(5) == (1.0, 0.0)
    assert compute_scales(7) == (1.0, 0.25)
    assert compute_scales(8) == (1.0, 0.5)
    assert compute_scales(10) == (1.0, 1.0)


def test_sger_warmup_schedule_rejects_invalid_epoch_ranges():
    compute_sger_warmup_scale, = _load_train_helpers(
        "compute_sger_warmup_scale")

    for start_epoch, end_epoch in ((-1, 6), (3, -1), (6, 3)):
        try:
            compute_sger_warmup_scale(0, start_epoch, end_epoch)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid warm-up epochs must raise ValueError")


def test_set_sger_warmup_state_updates_lite_model_and_loss_scale():
    compute_scale, set_state = _load_train_helpers(
        "compute_sger_warmup_scale", "set_sger_warmup_state")

    class FakeModel:
        use_sger_lite = True

        def set_sger_residual_scale(self, scale):
            self.scale = scale

    class Args:
        use_sger_lite = True
        sger_warmup_start_epoch = 3
        sger_warmup_end_epoch = 6

    model = FakeModel()
    args = Args()
    scale = set_state(model, args, 3)

    assert compute_scale(3, 3, 6) == 0.25
    assert scale == 0.25
    assert model.scale == 0.25
    assert args.current_sger_warmup_scale == 0.25


def test_set_sger_warmup_state_supports_wrapped_and_non_lite_models():
    _, set_state = _load_train_helpers(
        "compute_sger_warmup_scale", "set_sger_warmup_state")

    class FakeModel:
        def set_sger_residual_scale(self, scale):
            self.scale = scale

    class Wrapper:
        def __init__(self, module):
            self.module = module

    class Args:
        use_sger_lite = False
        sger_warmup_start_epoch = 3
        sger_warmup_end_epoch = 6

    model = FakeModel()
    args = Args()
    scale = set_state(Wrapper(model), args, 0)

    assert scale == 1.0
    assert model.scale == 1.0
    assert args.current_sger_warmup_scale == 1.0


def test_non_lite_optimizer_group_keeps_base_learning_rate():
    _, build_optimizer_param_groups = _load_train_helpers(
        "is_sger_lite_parameter", "build_optimizer_param_groups")
    model = CascadeMVSNet(
        ndepths=[8, 8, 8],
        depth_interals_ratio=[4, 2, 1],
        use_sger_lite=False,
    )

    groups = build_optimizer_param_groups(
        model, base_lr=0.001, backbone_lr_scale=0.1)

    assert len(groups) == 1
    assert groups[0]["lr"] == 0.001
    assert {id(param) for param in groups[0]["params"]} == {
        id(param) for param in model.parameters()}


def test_sger_lite_freeze_only_disables_backbone_parameters():
    _, _, set_sger_lite_freeze = _load_train_helpers(
        "is_sger_lite_parameter",
        "build_optimizer_param_groups",
        "set_sger_lite_freeze",
    )
    model = CascadeMVSNet(
        ndepths=[8, 8, 8],
        depth_interals_ratio=[4, 2, 1],
        use_sger_lite=True,
    )

    set_sger_lite_freeze(model, True)

    for name, param in model.named_parameters():
        expected_trainable = (
            name.startswith("normal_head.2.")
            or name.startswith("sger_blocks."))
        assert param.requires_grad == expected_trainable

    set_sger_lite_freeze(model, False)
    assert all(param.requires_grad for param in model.parameters())


def _load_test_helper(name):
    test_path = Path(__file__).resolve().parents[1] / "test.py"
    tree = ast.parse(test_path.read_text())
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    namespace = {"np": np}
    exec(compile(ast.Module(body=selected), str(test_path), "exec"), namespace)
    return namespace[name]


def test_residual_calibrated_confidence_uses_soft_floor():
    calibrate = _load_test_helper("residual_calibrated_confidence")
    confidence = np.array([0.2, 0.8, 1.2], dtype=np.float32)
    residual = np.array([0.0, 1.0, 100.0], dtype=np.float32)

    actual = calibrate(
        confidence, residual, alpha=0.25, confidence_floor=0.5)

    expected_scale = 0.5 + 0.5 * np.exp(-0.25 * residual)
    expected = np.clip(confidence * expected_scale, 0.0, 1.0)
    np.testing.assert_allclose(actual, expected.astype(np.float32))
    assert actual[0] == confidence[0]
    assert actual[2] >= 0.5 * min(confidence[2], 1.0)


def test_residual_calibrated_confidence_rejects_invalid_parameters():
    calibrate = _load_test_helper("residual_calibrated_confidence")

    for alpha, confidence_floor in ((-0.1, 0.5), (0.25, -0.1),
                                    (0.25, 1.1)):
        try:
            calibrate(
                np.ones(1, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                alpha=alpha,
                confidence_floor=confidence_floor,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "invalid calibration parameters must raise ValueError")


def test_resume_accepts_matching_or_legacy_sger_warmup_metadata():
    validate, = _load_train_helpers("validate_sger_warmup_checkpoint")

    class Args:
        sger_warmup_start_epoch = 3
        sger_warmup_end_epoch = 6

    validate({}, Args())
    validate({
        "sger_warmup_start_epoch": 3,
        "sger_warmup_end_epoch": 6,
    }, Args())


def test_resume_rejects_mismatched_sger_warmup_metadata():
    validate, = _load_train_helpers("validate_sger_warmup_checkpoint")

    class Args:
        sger_warmup_start_epoch = 3
        sger_warmup_end_epoch = 6

    try:
        validate({
            "sger_warmup_start_epoch": 2,
            "sger_warmup_end_epoch": 6,
        }, Args())
    except RuntimeError as error:
        assert "warm-up configuration mismatch" in str(error)
    else:
        raise AssertionError("resume must reject a changed warm-up schedule")


def test_train_and_test_entrypoints_expose_sger_configuration():
    root = Path(__file__).resolve().parents[1]
    train_source = (root / "train.py").read_text()
    test_source = (root / "test.py").read_text()
    gipuma_source = (root / "gipuma.py").read_text()

    model_flags = [
        "--use_sger",
        "--sger_share",
        "--sger_feature_channels",
        "--sger_hidden_channels",
        "--sger_gate_channels",
        "--sger_max_residual_ratio",
        "--detach_refined_feedback",
    ]
    for flag in model_flags:
        assert flag in train_source
        assert flag in test_source
    for flag in (
        "--raw_depth_loss_weight",
        "--refined_depth_loss_weight",
        "--residual_loss_weight",
    ):
        assert flag in train_source
    assert '"raw_depth_loss_weight": args.raw_depth_loss_weight' in train_source
    assert 'use_sger=args.use_sger' in train_source
    assert 'use_sger=args.use_sger' in test_source
    for image_key in ("depth_raw", "depth_residual", "geometry_gate", "region_a"):
        assert 'image_outputs["{}"]'.format(image_key) in train_source
    for source in (train_source, test_source):
        assert "--use_sger_lite" in source
        assert "--sger_max_residual_ratio', type=float, default=0.25" in source
    for flag in (
        "--freeze_backbone_epochs', type=int, default=8",
        "--raw_depth_loss_weight', type=float, default=1.0",
        "--refined_depth_loss_weight', type=float, default=1.0",
        "--residual_loss_weight', type=float, default=0.01",
        "--gate_loss_weight', type=float, default=0.0",
        "--safe_refine_loss_weight', type=float, default=0.1",
        "--safe_refine_margin', type=float, default=0.0",
        "--backbone_lr_scale', type=float, default=0.1",
        '"gate_loss_weight": args.gate_loss_weight',
        '"safe_refine_loss_weight": args.safe_refine_loss_weight',
        '"safe_refine_margin": args.safe_refine_margin',
        "--sger_warmup_start_epoch', type=int, default=7",
        "--sger_warmup_end_epoch', type=int, default=10",
        "--residual_target_loss_weight', type=float, default=0.05",
        "--gate_benefit_loss_weight', type=float, default=0.05",
        "--residual_target_ratio', type=float, default=0.25",
        "--benefit_margin_ratio', type=float, default=0.05",
        '"residual_target_loss_weight": args.residual_target_loss_weight',
        '"gate_benefit_loss_weight": args.gate_benefit_loss_weight',
        "'sger_warmup_start_epoch': args.sger_warmup_start_epoch",
        "'sger_warmup_end_epoch': args.sger_warmup_end_epoch",
    ):
        assert flag in train_source
    for export_hook in (
        "depth_est_raw",
        "confidence_residual_calibrated",
        "--fusion_depth_source",
        "--fusion_confidence_source",
        "get_fusion_depth_folder",
        "get_fusion_confidence_folder",
    ):
        assert export_hook in test_source
    assert "depth_folder=\"depth_est\"" in gipuma_source
    assert "confidence_folder=\"confidence\"" in gipuma_source
    assert "--fusion_depth_source', type=str, default='raw'" in test_source
    assert "--fusion_confidence_source', type=str, default='raw'" in test_source
    assert "--confidence_residual_alpha', type=float, default=0.25" in test_source
    assert "--confidence_residual_floor', type=float, default=0.5" in test_source
    assert "get_fusion_depth_folder()" in test_source
    assert "get_fusion_confidence_folder()" in test_source


if __name__ == "__main__":
    test_probability_volume_statistics_measure_uncertainty()
    test_probability_volume_statistics_handle_single_depth_bin()
    test_sger_block_returns_bounded_outputs()
    test_sger_block_separates_residual_proposal_and_benefit_gate()
    test_sger_regions_ignore_invalid_depth_and_are_disjoint()
    test_sger_sanitizes_nonfinite_depth_before_geometry_processing()
    test_sger_refined_depth_backpropagates_through_learned_path()
    test_cascade_sger_returns_raw_and_refined_depth_for_every_stage()
    test_cascade_sger_lite_refines_stage3_only()
    test_cascade_sger_lite_scales_effective_residual()
    test_cascade_sger_lite_refined_path_isolates_backbone_gradients()
    test_cascade_sger_residual_scale_rejects_invalid_values()
    test_cascade_disabled_path_has_no_sger_only_outputs()
    test_cascade_shared_sger_mode_runs_all_stages()
    test_cascade_uses_refined_depth_as_next_stage_sampling_center()
    test_sger_loss_supervises_raw_and_refined_depth_separately()
    test_bounded_residual_target_has_correct_direction_and_limit()
    test_residual_benefit_target_requires_margin_improvement()
    test_sger_benefit_supervision_reports_selective_refinement_metrics()
    test_sger_loss_regularizes_normalized_gated_residual()
    test_sger_gate_loss_uses_only_valid_pixels()
    test_safe_refine_loss_penalizes_only_regressions()
    test_safe_refine_loss_is_zero_for_improvement_and_applies_margin()
    test_sger_loss_scale_warms_all_refinement_terms()
    test_sger_loss_scale_logs_refined_improved_pixel_ratio()
    test_sger_loss_scale_rejects_invalid_values()
    test_sger_lite_optimizer_groups_cover_parameters_once()
    test_sger_warmup_schedule_matches_experiment13_epochs()
    test_experiment14_loss_schedule_separates_proposal_and_gate_training()
    test_sger_warmup_schedule_rejects_invalid_epoch_ranges()
    test_set_sger_warmup_state_updates_lite_model_and_loss_scale()
    test_set_sger_warmup_state_supports_wrapped_and_non_lite_models()
    test_non_lite_optimizer_group_keeps_base_learning_rate()
    test_sger_lite_freeze_only_disables_backbone_parameters()
    test_residual_calibrated_confidence_uses_soft_floor()
    test_residual_calibrated_confidence_rejects_invalid_parameters()
    test_resume_accepts_matching_or_legacy_sger_warmup_metadata()
    test_resume_rejects_mismatched_sger_warmup_metadata()
    test_train_and_test_entrypoints_expose_sger_configuration()
    print("SGER unit tests: PASS")
