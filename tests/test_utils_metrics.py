import torch

from utils import AbsDepthError_metrics, Thres_metrics


def test_abs_depth_error_metrics_handles_non_contiguous_inputs():
    depth_est = torch.arange(24, dtype=torch.float32).view(1, 4, 6).transpose(1, 2)
    depth_gt = torch.zeros_like(depth_est)
    mask = torch.ones_like(depth_est, dtype=torch.bool)

    value = AbsDepthError_metrics(depth_est, depth_gt, mask)

    assert torch.allclose(value, depth_est.abs().mean())


def test_abs_depth_error_metrics_returns_zero_for_empty_mask():
    depth_est = torch.ones(1, 2, 3)
    depth_gt = torch.zeros_like(depth_est)
    mask = torch.zeros_like(depth_est, dtype=torch.bool)

    value = AbsDepthError_metrics(depth_est, depth_gt, mask)

    assert value.item() == 0.0


def test_threshold_metrics_handles_non_contiguous_inputs():
    depth_est = torch.tensor([[[0.0, 3.0, 5.0],
                               [1.0, 4.0, 8.0]]]).transpose(1, 2)
    depth_gt = torch.zeros_like(depth_est)
    mask = torch.ones_like(depth_est, dtype=torch.bool)

    value = Thres_metrics(depth_est, depth_gt, mask, 4)

    assert torch.allclose(value, torch.tensor(2.0 / 6.0))
