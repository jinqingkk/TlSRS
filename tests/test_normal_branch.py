import torch
import torch.nn.functional as F

from models.module import (
    NormalHead,
    compute_normal_from_depth,
    build_smooth_mask,
    build_geometry_weight,
    build_dual_region_geometry,
    non_edge_depth_grad_mean,
    curvature_loss,
    soft_curvature_loss,
    dual_region_curvature_loss,
    depth_normal_consistency_loss,
    cas_mvsnet_loss,
)
from models.cas_mvsnet import CascadeMVSNet


def test_normal_head_outputs_unit_normals():
    torch.manual_seed(1)
    head = NormalHead(in_channels=10)
    x = torch.randn(2, 10, 8, 12)

    normal = head(x)

    assert normal.shape == (2, 3, 8, 12)
    norm = torch.norm(normal, p=2, dim=1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


def test_compute_normal_from_depth_flat_plane_points_forward():
    depth = torch.ones(1, 4, 5)
    intrinsics = torch.eye(3).unsqueeze(0)

    normal = compute_normal_from_depth(depth, intrinsics)

    assert normal.shape == (1, 3, 4, 5)
    expected = torch.zeros_like(normal)
    expected[:, 2] = 1.0
    assert torch.allclose(normal, expected, atol=1e-5)


def test_compute_normal_from_depth_uses_camera_space_intrinsics_for_plane_batch():
    batch_size, height, width = 2, 4, 5
    x = torch.arange(width).float().view(1, width).expand(height, width)
    y = torch.arange(height).float().view(height, 1).expand(height, width)
    intrinsics = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
    intrinsics[:, 0, 0] = torch.tensor([2.0, 4.0])
    intrinsics[:, 1, 1] = torch.tensor([3.0, 5.0])
    intrinsics[:, 0, 2] = torch.tensor([1.0, 2.0])
    intrinsics[:, 1, 2] = torch.tensor([1.5, 0.5])
    depth = torch.empty(batch_size, 1, height, width)

    expected = []
    for batch_idx in range(batch_size):
        fx = intrinsics[batch_idx, 0, 0]
        fy = intrinsics[batch_idx, 1, 1]
        cx = intrinsics[batch_idx, 0, 2]
        cy = intrinsics[batch_idx, 1, 2]
        a = torch.tensor(0.2 + 0.1 * batch_idx)
        b = torch.tensor(-0.15 + 0.05 * batch_idx)
        c = torch.tensor(1.5 + 0.2 * batch_idx)
        direction_x = (x - cx) / fx
        direction_y = (y - cy) / fy
        z = c / (1.0 - a * direction_x - b * direction_y)
        depth[batch_idx, 0] = z
        expected_normal = torch.tensor([-a, -b, 1.0])
        expected.append(expected_normal / torch.norm(expected_normal, p=2))
    expected = torch.stack(expected).view(batch_size, 3, 1, 1).expand(batch_size, 3, height, width)

    normal = compute_normal_from_depth(depth, intrinsics)

    assert normal.shape == (batch_size, 3, height, width)
    assert torch.allclose(normal, expected, atol=1e-4)


def test_build_smooth_mask_removes_edges_and_low_confidence():
    ref_img = torch.zeros(1, 3, 4, 6)
    ref_img[:, :, :, 3:] = 1.0
    valid_mask = torch.ones(1, 4, 6, dtype=torch.bool)
    confidence = torch.ones(1, 4, 6)
    confidence[:, :, 0] = 0.1

    smooth_mask = build_smooth_mask(
        ref_img,
        valid_mask,
        confidence,
        depth_normal_conf_threshold=0.8,
        edge_grad_threshold=0.05,
        target_size=(4, 6),
    )

    assert smooth_mask.shape == (1, 4, 6)
    assert not smooth_mask[:, :, 0].any()
    assert not smooth_mask[:, :, 2].any()
    assert not smooth_mask[:, :, 3].any()
    assert smooth_mask[:, :, 1].all()
    assert smooth_mask[:, :, 4:].all()


def test_build_smooth_mask_accepts_float_mask_and_4d_confidence_at_target_size():
    ref_img = torch.zeros(2, 3, 3, 4)
    valid_mask = torch.ones(2, 3, 4)
    valid_mask[0, 0, 0] = 0.0
    valid_mask[1, 2, 3] = 0.0
    confidence = torch.ones(2, 1, 3, 4)
    confidence[0, 0, 1, 1] = 0.2
    confidence[1, 0, 1, 2] = 0.2

    smooth_mask = build_smooth_mask(
        ref_img,
        valid_mask,
        confidence,
        depth_normal_conf_threshold=0.8,
        edge_grad_threshold=0.05,
        target_size=(3, 4),
    )

    assert smooth_mask.dtype == torch.bool
    assert smooth_mask.shape == (2, 3, 4)
    assert not smooth_mask[0, 0, 0]
    assert not smooth_mask[1, 2, 3]
    assert not smooth_mask[0, 1, 1]
    assert not smooth_mask[1, 1, 2]
    assert smooth_mask[0, 2, 2]
    assert smooth_mask[1, 0, 0]


def test_build_geometry_weight_softly_downweights_low_confidence_and_edges():
    ref_img = torch.zeros(1, 3, 4, 6)
    ref_img[:, :, :, 3:] = 1.0
    valid_mask = torch.ones(1, 4, 6, dtype=torch.bool)
    valid_mask[:, 0, 0] = False
    confidence = torch.ones(1, 1, 4, 6)
    confidence[:, :, :, 0] = 0.1

    geometry_weight, metrics = build_geometry_weight(
        ref_img,
        valid_mask,
        confidence,
        target_size=(4, 6),
        conf_mid=0.65,
        k_conf=10.0,
        edge_mid=0.25,
        k_edge=10.0,
        w_min=0.05,
    )

    assert geometry_weight.shape == (1, 4, 6)
    assert geometry_weight.dtype == ref_img.dtype
    assert geometry_weight[:, 0, 0].item() == 0.0
    assert geometry_weight[:, 1:, 0].min().item() > 0.0
    assert geometry_weight[:, 1:, 0].max().item() < 0.2
    assert geometry_weight[:, :, 2:4].max().item() < geometry_weight[:, :, 4:].min().item()
    assert geometry_weight[:, :, 4:].min().item() > 0.9
    assert metrics["geometry_weight_mean"].item() > 0.0
    assert metrics["geometry_weight_valid_mean"].item() >= 0.05
    assert metrics["high_weight_ratio"].item() > 0.0
    assert metrics["low_weight_ratio"].item() > 0.0


def test_build_dual_region_geometry_separates_hard_joint_and_soft_surface():
    ref_img = torch.zeros(1, 3, 3, 7)
    ref_img[:, :, :, 3:] = 1.0
    valid_mask = torch.ones(1, 3, 7, dtype=torch.bool)
    depth = torch.ones(1, 3, 7)
    depth[:, :, 3:] = 2.0
    depth[:, 1, 5] = 4.0
    confidence = torch.ones(1, 1, 3, 7)
    confidence[:, :, :, 0] = 0.1

    region_a, weight_b, metrics = build_dual_region_geometry(
        ref_img,
        valid_mask,
        depth,
        confidence,
        target_size=(3, 7),
        threshold_edge=0.25,
        threshold_depth=0.2,
        threshold_curv=0.3,
        conf_mid=0.65,
        k_conf=10.0,
        smooth_k=2.0,
    )

    assert region_a.dtype == torch.bool
    assert weight_b.dtype == ref_img.dtype
    assert region_a[:, :, 2].any()
    assert region_a[:, :, 3].any()
    assert region_a[:, 1, 5]
    assert torch.all(weight_b[region_a] == 0.0)
    assert weight_b[:, :, 0].max().item() < weight_b[:, :, 1].min().item()
    assert metrics["region_A_ratio"].item() > 0.0
    assert metrics["region_B_ratio"].item() > 0.0


def test_non_edge_depth_grad_mean_accepts_4d_depth():
    depth = torch.ones(2, 1, 3, 4)
    smooth_mask = torch.ones(2, 3, 4, dtype=torch.bool)

    value = non_edge_depth_grad_mean(depth, smooth_mask)

    assert value.item() < 1e-6


def test_depth_normal_consistency_loss_returns_metrics():
    depth = torch.ones(1, 4, 5)
    intrinsics = torch.eye(3).unsqueeze(0)
    normal = compute_normal_from_depth(depth, intrinsics)
    ref_img = torch.zeros(1, 3, 4, 5)
    mask = torch.ones(1, 4, 5, dtype=torch.bool)
    confidence = torch.ones(1, 4, 5)

    loss, metrics, smooth_mask = depth_normal_consistency_loss(
        normal,
        depth,
        intrinsics,
        ref_img,
        mask,
        confidence,
        depth_normal_conf_threshold=0.8,
        edge_grad_threshold=0.05,
    )

    assert loss.item() < 1e-5
    assert metrics["normal_depth_cos"].item() > 0.999
    assert metrics["smooth_mask_ratio"].item() > 0.999
    assert metrics["non_edge_depth_grad_mean"].item() < 1e-5
    assert smooth_mask.shape == (1, 4, 5)


def test_depth_normal_consistency_loss_accepts_4d_depth_confidence_and_batch():
    depth = torch.ones(2, 1, 4, 5, requires_grad=True)
    intrinsics = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    normal = compute_normal_from_depth(depth, intrinsics)
    ref_img = torch.zeros(2, 3, 4, 5)
    mask = torch.ones(2, 4, 5)
    confidence = torch.ones(2, 1, 4, 5)

    loss, metrics, smooth_mask = depth_normal_consistency_loss(
        normal,
        depth,
        intrinsics,
        ref_img,
        mask,
        confidence,
        depth_normal_conf_threshold=0.8,
        edge_grad_threshold=0.05,
    )

    assert loss.item() < 1e-5
    assert smooth_mask.shape == (2, 4, 5)
    assert metrics["normal_depth_cos"].item() > 0.999
    assert not metrics["non_edge_depth_grad_mean"].requires_grad


def test_depth_normal_consistency_loss_scales_intrinsics_for_lower_resolution_depth():
    source_size = (8, 10)
    target_size = (4, 5)
    full_intrinsics = torch.eye(3).unsqueeze(0)
    full_intrinsics[:, 0, 0] = 8.0
    full_intrinsics[:, 1, 1] = 10.0
    full_intrinsics[:, 0, 2] = 4.0
    full_intrinsics[:, 1, 2] = 3.0
    scaled_intrinsics = full_intrinsics.clone()
    scaled_intrinsics[:, 0, :] *= float(target_size[1]) / float(source_size[1])
    scaled_intrinsics[:, 1, :] *= float(target_size[0]) / float(source_size[0])
    x = torch.arange(target_size[1]).float().view(1, target_size[1]).expand(target_size)
    y = torch.arange(target_size[0]).float().view(target_size[0], 1).expand(target_size)
    a = torch.tensor(0.25)
    b = torch.tensor(-0.1)
    c = torch.tensor(1.7)
    direction_x = (x - scaled_intrinsics[0, 0, 2]) / scaled_intrinsics[0, 0, 0]
    direction_y = (y - scaled_intrinsics[0, 1, 2]) / scaled_intrinsics[0, 1, 1]
    depth = (c / (1.0 - a * direction_x - b * direction_y)).unsqueeze(0)
    expected_normal = torch.tensor([-a, -b, 1.0])
    expected_normal = expected_normal / torch.norm(expected_normal, p=2)
    normal = expected_normal.view(1, 3, 1, 1).expand(1, 3, target_size[0], target_size[1])
    ref_img = torch.zeros(1, 3, source_size[0], source_size[1])
    mask = torch.ones(1, target_size[0], target_size[1], dtype=torch.bool)
    confidence = torch.ones(1, target_size[0], target_size[1])

    loss, metrics, smooth_mask = depth_normal_consistency_loss(
        normal,
        depth,
        full_intrinsics,
        ref_img,
        mask,
        confidence,
        depth_normal_conf_threshold=0.8,
        edge_grad_threshold=0.05,
    )

    assert smooth_mask.shape == (1, target_size[0], target_size[1])
    assert loss.item() < 1e-5
    assert metrics["normal_depth_cos"].item() > 0.999


def test_depth_normal_consistency_loss_backpropagates_to_normals_and_depth():
    torch.manual_seed(4)
    depth = 1.0 + torch.rand(1, 4, 5)
    depth.requires_grad_()
    normal = torch.zeros(1, 3, 4, 5)
    normal[:, 2] = 1.0
    normal.requires_grad_()
    intrinsics = torch.eye(3).unsqueeze(0)
    ref_img = torch.zeros(1, 3, 4, 5)
    mask = torch.ones(1, 4, 5, dtype=torch.bool)
    confidence = torch.ones(1, 4, 5)

    loss, _, _ = depth_normal_consistency_loss(
        normal,
        depth,
        intrinsics,
        ref_img,
        mask,
        confidence,
        depth_normal_conf_threshold=0.8,
        edge_grad_threshold=0.05,
    )
    loss.backward()

    assert normal.grad is not None
    assert depth.grad is not None
    assert normal.grad.abs().sum().item() > 0.0
    assert depth.grad.abs().sum().item() > 0.0


from models.cas_mvsnet import CascadeMVSNet


def _identity_proj(batch, views):
    proj = torch.eye(4).view(1, 1, 1, 4, 4).repeat(batch, views, 2, 1, 1)
    return proj


def _unpack_loss_result(result):
    return result[0], result[1], result[-1]


def test_cascade_outputs_normal_branch_for_every_stage():
    torch.manual_seed(2)
    model = CascadeMVSNet(ndepths=[8, 8, 8], depth_interals_ratio=[4, 2, 1])
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

    expected_shapes = {
        "stage1": (1, 3, 8, 8),
        "stage2": (1, 3, 16, 16),
        "stage3": (1, 3, 32, 32),
    }
    for stage_key, expected_shape in expected_shapes.items():
        assert "normal" in outputs[stage_key]
        assert outputs[stage_key]["normal"].shape == expected_shape
        norm = torch.norm(outputs[stage_key]["normal"], p=2, dim=1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-4)
    assert "normal" in outputs
    assert outputs["normal"].shape == expected_shapes["stage3"]


def test_cascade_two_stage_model_outputs_normal_for_active_stages():
    torch.manual_seed(3)
    model = CascadeMVSNet(ndepths=[8, 8], depth_interals_ratio=[4, 2])
    model.eval()
    imgs = torch.rand(1, 3, 3, 32, 32)
    depth_values = torch.linspace(1.0, 2.0, 8).view(1, 8)
    proj_matrices = {
        "stage1": _identity_proj(1, 3),
        "stage2": _identity_proj(1, 3),
    }

    with torch.no_grad():
        outputs = model(imgs, proj_matrices, depth_values)

    assert "normal" in outputs["stage1"]
    assert "normal" in outputs["stage2"]
    assert outputs["stage1"]["normal"].shape == (1, 3, 8, 8)
    assert outputs["stage2"]["normal"].shape == (1, 3, 16, 16)
    assert "normal" in outputs
    assert outputs["normal"].shape == (1, 3, 16, 16)


def test_cascade_model_loads_depth_only_checkpoint_with_missing_normal_head():
    model = CascadeMVSNet(ndepths=[8, 8, 8], depth_interals_ratio=[4, 2, 1])
    state = model.state_dict()
    depth_only_state = {k: v for k, v in state.items() if not k.startswith("normal_head")}

    missing_keys, unexpected_keys = model.load_state_dict(depth_only_state, strict=False)

    assert unexpected_keys == []
    assert any(k.startswith("normal_head") for k in missing_keys)


def test_cas_mvsnet_loss_includes_depth_normal_metrics():
    depth = torch.ones(1, 4, 5)
    intrinsics = torch.eye(3).view(1, 1, 3, 3)
    proj = torch.eye(4).view(1, 1, 1, 4, 4).repeat(1, 1, 2, 1, 1)
    proj[:, 0, 1, :3, :3] = intrinsics[:, 0]
    normal = compute_normal_from_depth(depth, intrinsics[:, 0])
    inputs = {
        "stage1": {"depth": torch.ones(1, 1, 2) * 1.0},
        "stage2": {"depth": torch.ones(1, 2, 3) * 1.0},
        "stage3": {
            "depth": depth,
            "photometric_confidence": torch.ones(1, 4, 5),
            "normal": normal,
        },
        "depth": depth,
        "photometric_confidence": torch.ones(1, 4, 5),
        "normal": normal,
    }
    depth_gt_ms = {
        "stage1": torch.ones(1, 1, 2),
        "stage2": torch.ones(1, 2, 3),
        "stage3": torch.ones(1, 4, 5),
    }
    mask_ms = {
        "stage1": torch.ones(1, 1, 2),
        "stage2": torch.ones(1, 2, 3),
        "stage3": torch.ones(1, 4, 5),
    }
    proj_matrices = {"stage3": proj}
    imgs = torch.zeros(1, 1, 3, 4, 5)

    total_loss, depth_loss, extra = _unpack_loss_result(cas_mvsnet_loss(
        inputs,
        depth_gt_ms,
        mask_ms,
        imgs=imgs,
        proj_matrices=proj_matrices,
        dlossw=[0.5, 1.0, 2.0],
        depth_normal_loss_weight=0.03,
        depth_normal_conf_threshold=0.8,
        edge_grad_threshold=0.05,
        return_extra=True,
    ))

    assert total_loss.item() < 1e-5
    assert depth_loss.item() < 1e-5
    assert extra["depth_normal_loss"].item() < 1e-5
    assert extra["normal_depth_cos"].item() > 0.999
    assert extra["smooth_mask_ratio"].item() > 0.999


def test_cas_mvsnet_loss_skips_depth_normal_without_stage3_projection():
    depth = torch.ones(1, 4, 5)
    intrinsics = torch.eye(3).view(1, 1, 3, 3)
    normal = compute_normal_from_depth(depth, intrinsics[:, 0])
    inputs = {
        "stage1": {"depth": torch.ones(1, 1, 2) * 1.0},
        "stage2": {"depth": torch.ones(1, 2, 3) * 1.0},
        "stage3": {
            "depth": depth,
            "photometric_confidence": torch.ones(1, 4, 5),
            "normal": normal,
        },
        "depth": depth,
        "photometric_confidence": torch.ones(1, 4, 5),
        "normal": normal,
    }
    depth_gt_ms = {
        "stage1": torch.ones(1, 1, 2),
        "stage2": torch.ones(1, 2, 3),
        "stage3": torch.ones(1, 4, 5),
    }
    mask_ms = {
        "stage1": torch.ones(1, 1, 2),
        "stage2": torch.ones(1, 2, 3),
        "stage3": torch.ones(1, 4, 5),
    }
    imgs = torch.zeros(1, 1, 3, 4, 5)

    total_loss, depth_loss, extra = _unpack_loss_result(cas_mvsnet_loss(
        inputs,
        depth_gt_ms,
        mask_ms,
        imgs=imgs,
        proj_matrices={},
        dlossw=[0.5, 1.0, 2.0],
        depth_normal_loss_weight=0.03,
        depth_normal_conf_threshold=0.8,
        edge_grad_threshold=0.05,
        return_extra=True,
    ))

    assert total_loss.item() < 1e-5
    assert depth_loss.item() < 1e-5
    assert "depth_normal_loss" not in extra


def test_cas_mvsnet_loss_uses_sorted_numeric_stage_keys_only():
    stage1_depth = torch.ones(1, 1, 2) * 1.0
    stage2_depth = torch.ones(1, 2, 3) * 1.0
    stage3_depth = torch.ones(1, 4, 5) * 1.5
    inputs = {
        "stage2": {"depth": stage2_depth},
        "not_a_stage_key": {},
        "stage3": {"depth": stage3_depth},
        "stage1": {"depth": stage1_depth},
    }
    depth_gt_ms = {
        "stage1": torch.ones(1, 1, 2),
        "stage2": torch.ones(1, 2, 3),
        "stage3": torch.ones(1, 4, 5),
    }
    mask_ms = {
        "stage1": torch.ones(1, 1, 2),
        "stage2": torch.ones(1, 2, 3),
        "stage3": torch.ones(1, 4, 5),
    }

    _, depth_loss, _ = _unpack_loss_result(cas_mvsnet_loss(
        inputs,
        depth_gt_ms,
        mask_ms,
        return_extra=True,
    ))

    expected_stage3_loss = F.smooth_l1_loss(
        stage3_depth.view(-1),
        depth_gt_ms["stage3"].view(-1),
        reduction='mean',
    )
    assert torch.allclose(depth_loss, expected_stage3_loss)


def test_soft_curvature_loss_uses_weighted_second_order_stencil():
    depth = torch.tensor([[[1.0, 1.0, 4.0, 1.0, 1.0]]])
    weight = torch.tensor([[[1.0, 0.5, 0.5, 1.0, 1.0]]])

    loss = soft_curvature_loss(depth, weight)

    depth_mean = (depth * weight).sum() / weight.sum()
    depth_norm = depth / depth_mean
    curv_x = (depth_norm[:, :, 2:] - 2 * depth_norm[:, :, 1:-1] + depth_norm[:, :, :-2]).abs()
    stencil_weight = torch.min(torch.min(weight[:, :, 2:], weight[:, :, 1:-1]), weight[:, :, :-2])
    expected = (curv_x * stencil_weight).sum() / (stencil_weight.sum() + 1e-6)
    assert torch.allclose(loss, expected)


def test_soft_geometry_weight_does_not_backpropagate_through_confidence():
    depth = torch.tensor([[[1.0, 1.0, 4.0, 1.0, 2.0]]], requires_grad=True)
    confidence = torch.tensor([[[0.2, 0.4, 0.6, 0.8, 1.0]]], requires_grad=True)
    geometry_weight, _ = build_geometry_weight(
        torch.zeros(1, 3, 1, 5),
        torch.ones(1, 1, 5),
        confidence,
        target_size=(1, 5),
    )

    loss = soft_curvature_loss(depth, geometry_weight)
    loss.backward()

    assert depth.grad is not None
    assert depth.grad.abs().sum().item() > 0.0
    assert confidence.grad is None


def test_dual_region_curvature_loss_combines_hard_and_soft_regions():
    depth = torch.tensor([[[1.0, 1.0, 4.0, 1.0, 2.0]]])
    region_a = torch.tensor([[[False, True, True, True, False]]])
    weight_b = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.5]]])

    total, loss_a, loss_b = dual_region_curvature_loss(
        depth,
        region_a,
        weight_b,
        lambda_a=1.5,
        lambda_b=1.0,
    )

    depth_mean = depth.mean()
    depth_norm = depth / depth_mean
    curv_x = (depth_norm[:, :, 2:] - 2 * depth_norm[:, :, 1:-1] + depth_norm[:, :, :-2]).abs()
    region_a_x = region_a[:, :, 2:] & region_a[:, :, 1:-1] & region_a[:, :, :-2]
    weight_b_x = torch.min(torch.min(weight_b[:, :, 2:], weight_b[:, :, 1:-1]), weight_b[:, :, :-2])
    expected_a = (curv_x * region_a_x.float()).sum() / (region_a_x.float().sum() + 1e-6)
    expected_b = (curv_x * weight_b_x).sum() / (weight_b_x.sum() + 1e-6)

    assert torch.allclose(loss_a, expected_a)
    assert torch.allclose(loss_b, expected_b)
    assert torch.allclose(total, 1.5 * expected_a + expected_b)


def test_cas_mvsnet_loss_depth_normal_branch_backpropagates_to_depth_and_normal():
    torch.manual_seed(5)
    depth = 1.0 + torch.rand(1, 4, 5)
    depth.requires_grad_()
    normal = torch.zeros(1, 3, 4, 5)
    normal[:, 2] = 1.0
    normal.requires_grad_()
    proj = torch.eye(4).view(1, 1, 1, 4, 4).repeat(1, 1, 2, 1, 1)
    inputs = {
        "stage1": {"depth": torch.ones(1, 1, 2)},
        "stage2": {"depth": torch.ones(1, 2, 3)},
        "stage3": {
            "depth": depth,
            "photometric_confidence": torch.ones(1, 4, 5),
            "normal": normal,
        },
    }
    depth_gt_ms = {
        "stage1": torch.ones(1, 1, 2),
        "stage2": torch.ones(1, 2, 3),
        "stage3": depth.detach().clone(),
    }
    mask_ms = {
        "stage1": torch.ones(1, 1, 2),
        "stage2": torch.ones(1, 2, 3),
        "stage3": torch.ones(1, 4, 5),
    }
    imgs = torch.zeros(1, 1, 3, 4, 5)

    total_loss, depth_loss, extra = _unpack_loss_result(cas_mvsnet_loss(
        inputs,
        depth_gt_ms,
        mask_ms,
        imgs=imgs,
        proj_matrices={"stage3": proj},
        dlossw=[0.0, 0.0, 0.0],
        depth_normal_loss_weight=1.0,
        depth_normal_conf_threshold=0.8,
        edge_grad_threshold=0.05,
        return_extra=True,
    ))
    total_loss.backward()

    assert depth_loss.item() < 1e-6
    assert extra["depth_normal_loss"].item() > 0.0
    assert normal.grad is not None
    assert depth.grad is not None
    assert normal.grad.abs().sum().item() > 0.0
    assert depth.grad.abs().sum().item() > 0.0


def test_cas_mvsnet_loss_uses_current_stage_projection_for_depth_normal():
    depth = torch.ones(1, 2, 3)
    intrinsics = torch.eye(3).view(1, 1, 3, 3)
    normal = compute_normal_from_depth(depth, intrinsics[:, 0])
    proj = torch.eye(4).view(1, 1, 1, 4, 4).repeat(1, 1, 2, 1, 1)
    proj[:, 0, 1, :3, :3] = intrinsics[:, 0]
    inputs = {
        "stage1": {
            "depth": depth,
            "photometric_confidence": torch.ones(1, 2, 3),
            "normal": normal,
        },
    }
    depth_gt_ms = {"stage1": depth.detach().clone()}
    mask_ms = {"stage1": torch.ones(1, 2, 3)}
    imgs = torch.zeros(1, 1, 3, 2, 3)

    result = cas_mvsnet_loss(
        inputs,
        depth_gt_ms,
        mask_ms,
        imgs=imgs,
        proj_matrices={"stage1": proj},
        dlossw=[0.0],
        depth_normal_loss_weight=1.0,
        depth_normal_conf_threshold=0.8,
        edge_grad_threshold=0.05,
        return_extra=True,
    )

    extra = result[-1]
    assert "depth_normal_loss" in extra
    assert extra["depth_normal_loss"].item() < 1e-5


def test_cas_mvsnet_loss_uses_soft_geometry_weight_for_curvature_metrics():
    depth = torch.tensor([[[1.0, 1.0, 4.0, 1.0, 1.0]]])
    confidence = torch.ones(1, 1, 5)
    confidence[:, :, 2] = 0.1
    inputs = {
        "stage1": {
            "depth": depth,
            "photometric_confidence": confidence,
        },
    }
    depth_gt_ms = {"stage1": depth.clone()}
    mask_ms = {"stage1": torch.ones(1, 1, 5)}
    imgs = torch.zeros(1, 1, 3, 1, 5)

    _, _, extra = _unpack_loss_result(cas_mvsnet_loss(
        inputs,
        depth_gt_ms,
        mask_ms,
        imgs=imgs,
        dlossw=[0.0],
        curv_loss_weight=1.0,
        depth_normal_conf_threshold=0.8,
        edge_grad_threshold=0.05,
        return_extra=True,
    ))

    assert extra["curv_loss"].item() > 0.0
    expected_raw = curvature_loss(depth, mask_ms["stage1"] > 0.5)
    assert torch.allclose(extra["stage1/curvature_loss_raw"], expected_raw)
    assert torch.allclose(extra["curvature_loss_raw"], 0.5 * expected_raw)
    assert torch.allclose(extra["curvature_loss_weighted"], extra["curv_loss"])
    assert extra["geometry_weight_mean"].item() > 0.0
    assert extra["geometry_weight_valid_mean"].item() >= 0.05
    assert extra["high_weight_ratio"].item() > 0.0
    assert extra["low_weight_ratio"].item() > 0.0
    assert "stage1/geometry_weight_mean" in extra


def test_cas_mvsnet_loss_uses_requested_multistage_curvature_weights():
    stage_depths = {
        "stage1": torch.tensor([[[1.0, 1.0, 2.0, 1.0, 1.0]]]),
        "stage2": torch.tensor([[[1.0, 1.0, 3.0, 1.0, 1.0]]]),
        "stage3": torch.tensor([[[1.0, 1.0, 4.0, 1.0, 1.0]]]),
    }
    inputs = {
        stage_key: {
            "depth": depth,
            "photometric_confidence": torch.ones_like(depth),
        }
        for stage_key, depth in stage_depths.items()
    }
    depth_gt_ms = {stage_key: depth.clone() for stage_key, depth in stage_depths.items()}
    mask_ms = {stage_key: torch.ones_like(depth) for stage_key, depth in stage_depths.items()}
    imgs = torch.zeros(1, 1, 3, 1, 5)

    total_loss, _, extra = _unpack_loss_result(cas_mvsnet_loss(
        inputs,
        depth_gt_ms,
        mask_ms,
        imgs=imgs,
        dlossw=[0.0, 0.0, 0.0],
        curv_loss_weight=1.0,
        return_extra=True,
    ))

    expected = (
        0.5 * extra["stage1/curvature_loss_weighted"]
        + 1.0 * extra["stage2/curvature_loss_weighted"]
        + 1.5 * extra["stage3/curvature_loss_weighted"]
    )
    assert torch.allclose(extra["curvature_loss_weighted"], expected)
    assert torch.allclose(extra["curv_loss"], expected)
    assert torch.allclose(total_loss, expected)


def test_cas_mvsnet_loss_uses_dual_region_curvature_when_enabled():
    stage_depths = {
        "stage1": torch.tensor([[[1.0, 1.0, 3.0, 1.0, 1.0]]]),
        "stage2": torch.tensor([[[1.0, 1.0, 4.0, 1.0, 1.0]]]),
        "stage3": torch.tensor([[[1.0, 1.0, 5.0, 1.0, 1.0]]]),
    }
    inputs = {
        stage_key: {
            "depth": depth,
            "photometric_confidence": torch.ones_like(depth),
        }
        for stage_key, depth in stage_depths.items()
    }
    depth_gt_ms = {stage_key: torch.ones_like(depth) for stage_key, depth in stage_depths.items()}
    mask_ms = {stage_key: torch.ones_like(depth) for stage_key, depth in stage_depths.items()}
    imgs = torch.zeros(1, 1, 3, 1, 5)
    imgs[:, :, :, :, 2:] = 1.0

    total_loss, _, extra = _unpack_loss_result(cas_mvsnet_loss(
        inputs,
        depth_gt_ms,
        mask_ms,
        imgs=imgs,
        dlossw=[0.0, 0.0, 0.0],
        curv_loss_weight=1.0,
        use_dual_region_curvature=True,
        region_lambda_a=1.5,
        region_lambda_b=1.0,
        region_edge_threshold=0.25,
        region_depth_threshold=0.2,
        region_curv_threshold=0.2,
        return_extra=True,
    ))

    expected = (
        0.5 * extra["stage1/curv_loss_total"]
        + 1.0 * extra["stage2/curv_loss_total"]
        + 1.5 * extra["stage3/curv_loss_total"]
    )
    assert torch.allclose(extra["curv_loss_total"], expected)
    assert torch.allclose(extra["curv_loss"], expected)
    assert torch.allclose(total_loss, expected)
    assert extra["curv_loss_A"].item() > 0.0
    assert extra["curv_loss_B"].item() >= 0.0
    assert extra["region_A_ratio"].item() > 0.0
    assert extra["region_B_ratio"].item() > 0.0
    assert "acc_region_A" in extra
    assert "acc_region_B" in extra
