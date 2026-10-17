"""CPU integration: production cascade/loss with two fixed synthetic batches."""
import json
import os
import tempfile
from unittest.mock import patch

import torch
from gradient_audit import main


class SyntheticDTU(torch.utils.data.Dataset):
    metas = [('synthetic', i) for i in range(6)]

    def __init__(self, *args):
        pass

    def __len__(self):
        return 6

    def __getitem__(self, index):
        camera = torch.eye(4).repeat(3, 2, 1, 1)
        return dict(imgs=torch.full((3, 3, 32, 32), 0.1 * index),
                    proj_matrices={k: camera.clone() for k in ('stage1', 'stage2', 'stage3')},
                    depth_values=torch.linspace(1, 10, 192),
                    depth={k: torch.ones(s, s) * 4 for k, s in [('stage1', 8), ('stage2', 16), ('stage3', 32)]},
                    mask={k: torch.ones(s, s) for k, s in [('stage1', 8), ('stage2', 16), ('stage3', 32)]})


def test_fixed_batches():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory:
        with patch('gradient_audit.find_dataset_def', return_value=SyntheticDTU):
            main(['--trainpath', directory, '--trainlist', 'lists/dtu/train.txt',
                  '--output_json', os.path.join(directory, 'audit.json'),
                  '--audit_epoch', '12', '--device', 'cpu', '--batch_size', '2',
                  '--batch_indices', '0,2', '--ndepths', '8,8,8'])
        for batch, indices in [(0, [0, 1]), (2, [4, 5])]:
            with open(os.path.join(directory, 'audit_batch{:04d}.json'.format(batch))) as handle:
                report = json.load(handle)
            assert report['metadata']['dataset_indices'] == indices
            assert len(report['cases']) == 15
            for case in report['cases']:
                stats = case['loss_extras']['residual_diagnostics']['stage3']
                assert stats['captured']
                assert stats['activations']['output.input']['shape'][0] == 2
                assert 'output.bias' in case['residual_parameter_gradients']['sger_blocks.0']
            assert report['cases'][0]['gradient_summary']['all_finite']
        assert not os.path.exists(os.path.join(directory, 'audit.json'))


if __name__ == '__main__':
    test_fixed_batches()
    print('Residual multi-batch integration: PASS')
