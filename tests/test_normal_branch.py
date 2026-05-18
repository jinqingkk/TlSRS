import torch

from models.module import (
    NormalHead,
    compute_normal_from_depth,
    build_smooth_mask,
    depth_normal_consistency_loss,
)


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
