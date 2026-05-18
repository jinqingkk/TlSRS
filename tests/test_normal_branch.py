import torch
import torch.nn.functional as F

from models.module import (
    NormalHead,
    compute_normal_from_depth,
    build_smooth_mask,
    non_edge_depth_grad_mean,
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


def test_cascade_outputs_stage3_normal_branch():
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

    assert "normal" in outputs
    assert "normal" in outputs["stage3"]
    assert outputs["normal"].shape == (1, 3, 32, 32)
    norm = torch.norm(outputs["normal"], p=2, dim=1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-4)


def test_cascade_two_stage_model_does_not_output_normal_branch():
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

    assert "normal" not in outputs
    assert "normal" not in outputs["stage2"]


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

    total_loss, depth_loss, extra = cas_mvsnet_loss(
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
    )

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

    total_loss, depth_loss, extra = cas_mvsnet_loss(
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
    )

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

    _, depth_loss, _ = cas_mvsnet_loss(
        inputs,
        depth_gt_ms,
        mask_ms,
        return_extra=True,
    )

    expected_stage3_loss = F.smooth_l1_loss(
        stage3_depth.view(-1),
        depth_gt_ms["stage3"].view(-1),
        reduction='mean',
    )
    assert torch.allclose(depth_loss, expected_stage3_loss)


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

    total_loss, depth_loss, extra = cas_mvsnet_loss(
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
    )
    total_loss.backward()

    assert depth_loss.item() < 1e-6
    assert extra["depth_normal_loss"].item() > 0.0
    assert normal.grad is not None
    assert depth.grad is not None
    assert normal.grad.abs().sum().item() > 0.0
    assert depth.grad.abs().sum().item() > 0.0
