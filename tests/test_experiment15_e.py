import subprocess
from pathlib import Path

import numpy as np
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
import torch

from experiment15_e_amplitude import interpolate
from experiment15_e_diagnose import statistics, pool
from gipuma import depth_map_fusion
from models.sger_refinement import SGERBlock, GeometryCueExtractor
from experiment15_e_reprojection import reprojection


def check_fusion(tmp_path):
    old = tmp_path / 'consistencyCheck-old'
    old.mkdir()
    (old/'final3d_model.ply').write_text('old')
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    patcher = patch.object(subprocess, 'run', fail)
    patcher.start()
    with unittest.TestCase().assertRaises(subprocess.CalledProcessError):
        depth_map_fusion(str(tmp_path), '/fake executable', .25, 4)
    patcher.stop()
    patcher = patch.object(subprocess, 'run', lambda *a, **k: None)
    patcher.start()
    with unittest.TestCase().assertRaisesRegex(RuntimeError, 'new nonempty'):
        depth_map_fusion(str(tmp_path), '/fake executable', .25, 4)
    def success(command, **kwargs):
        assert command[0] == '/fake executable'
        new = tmp_path/'consistencyCheck-new'
        new.mkdir()
        (new/'final3d_model.ply').write_text('new')
    patcher.stop()
    patcher = patch.object(subprocess, 'run', success)
    patcher.start()
    assert depth_map_fusion(str(tmp_path), '/fake executable', .25, 4).endswith('consistencyCheck-new/final3d_model.ply')
    patcher.stop()


def test_fixed_amplitudes_and_harm_statistics():
    raw = np.array([[10, 10, 10, 10]], np.float32)
    gt = np.array([[10.1, 12, 8, 8]], np.float32)
    proposal = np.array([[1, 1, -1, 1]], np.float32)
    refined = raw + proposal
    assert np.array_equal(interpolate(raw, refined, 0), raw)
    assert np.array_equal(interpolate(raw, refined, 1), refined)
    assert np.allclose(interpolate(raw, refined, .25), raw + .25*proposal)
    stats = pool([statistics(raw, refined, proposal, np.ones_like(raw), gt,
                             np.ones_like(raw), np.ones_like(raw), np.zeros_like(raw))])
    metrics = stats['regions']['all']['means']
    assert metrics['correct_direction_overshoot'] == .25
    assert abs(metrics['harm'] - .45) < 1e-6
    assert metrics['benefit'] == .5
    assert stats['balanced_direction_accuracy'] == .75


def test_reprojection_plane_translation_and_unsupported_points():
    depth = np.full((8, 12), 10, np.float32)
    intrinsics = np.array([[10, 0, 5], [0, 10, 3], [0, 0, 1]], np.float32)
    reference = np.eye(4, dtype=np.float32)
    source = reference.copy()
    source[0, 3] = 1
    support, pixels, relative = reprojection(depth, (intrinsics, reference), depth, (intrinsics, source))
    assert support.sum() > 20
    assert pixels[support].max() < 1e-4
    assert relative[support].max() < 1e-5
    support, _, _ = reprojection(depth, (intrinsics, reference), np.zeros_like(depth), (intrinsics, source))
    assert not support.any()
    source[2, 3] = -20
    support, _, _ = reprojection(depth, (intrinsics, reference), depth, (intrinsics, source))
    assert not support.any()


def test_sample_cues_and_gate_are_independent_of_batch_composition():
    torch.manual_seed(7)
    depth = torch.rand(2, 8, 12) + 10
    depth[1] *= 3
    depth[0, 0, 0] = 0
    normal = torch.nn.functional.normalize(torch.randn(2, 3, 8, 12), dim=1)
    confidence, image = torch.ones_like(depth)*.9, torch.rand(2, 3, 8, 12)
    intrinsics, feature = torch.eye(3).repeat(2, 1, 1), torch.rand(2, 8, 8, 12)
    inputs = (depth, normal, confidence, image, intrinsics, feature)
    block = SGERBlock(8, depth_normalization='sample').eval()
    with torch.no_grad():
        block.residual.output.weight.normal_(0, .01)
    batch = block(*inputs, depth_interval=1)
    single = block(*[x[:1] for x in inputs], depth_interval=1)
    for key in ('depth_refined', 'raw_depth_residual', 'benefit_gate', 'region_b_weight'):
        assert torch.allclose(batch[key][:1], single[key], atol=2e-6, rtol=1e-6), key
    legacy = GeometryCueExtractor('batch')
    assert not torch.allclose(legacy(*inputs[:5])['cue_tensor'][:1],
                              legacy(*[x[:1] for x in inputs[:5]])['cue_tensor'])
    sample = GeometryCueExtractor('sample')
    empty = sample(torch.zeros_like(depth), normal, confidence, image, intrinsics)
    assert torch.isfinite(empty['cue_tensor']).all()


def test_normalization_preserves_parameter_contract_and_learnable_gradients():
    old, new = SGERBlock(8), SGERBlock(8, depth_normalization='sample')
    new.load_state_dict(old.state_dict(), strict=True)
    depth = torch.ones(2, 6, 8)*10
    normal = torch.randn(2, 3, 6, 8, requires_grad=True)
    feature = torch.randn(2, 8, 6, 8, requires_grad=True)
    # Nonzero head is necessary for body/gate gradients to be active.
    with torch.no_grad():
        new.residual.output.weight.normal_(0, .01)
        new.residual.output.bias.fill_(.2)
    result = new(depth, normal, torch.ones_like(depth)*.9, torch.rand(2, 3, 6, 8),
                 torch.eye(3).repeat(2, 1, 1), feature, depth_interval=1)
    result['depth_refined'].sum().backward()
    assert normal.grad is not None and torch.isfinite(normal.grad).all()
    for name, parameter in new.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert new.residual.output.weight.grad.abs().sum() > 0
    assert result['depth_residual'].abs().max() <= new.max_residual_ratio


def test_fusion_rejects_failure_and_stale_ply():
    with TemporaryDirectory() as directory:
        try:
            check_fusion(Path(directory))
        finally:
            patch.stopall()


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
                               for name, value in sorted(globals().items())
                               if name.startswith('test_'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
