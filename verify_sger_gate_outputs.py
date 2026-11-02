"""Verify completed paired gate validation and fusion artifacts without rerunning models."""
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from datasets.data_io import read_pfm
from evaluation_record import sha256, write_json


def verify(validation_path, test_root, output_path):
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError('Use a new verification output path')
    validation = json.loads(Path(validation_path).read_text())
    if validation['status'] != 'complete' or not validation['runs']:
        raise ValueError('Validation is incomplete')
    validation_rows, protected = [], None
    for checkpoint, run in validation['runs'].items():
        if sha256(checkpoint) != run['checkpoint_sha256']:
            raise ValueError('Checkpoint changed: ' + checkpoint)
        result = run['result']
        hashes = result['protected_output_sha256']
        if len(hashes) != len(validation['validation_metas']):
            raise ValueError('Validation sample count mismatch')
        if protected is not None and hashes != protected:
            raise ValueError('Protected outputs differ across checkpoints')
        protected = hashes
        means = result['aggregate']['regions']['all']['means']
        validation_rows.append(dict(checkpoint=checkpoint, **means))

    root = Path(test_root)
    paired = root / 'paired'
    inference = json.loads((paired / 'inference_manifest.json').read_text())
    amplitude = json.loads((root / 'fusion/amplitude_manifest.json').read_text())
    if (inference['status'] != 'complete' or not inference['paired_forward']
            or inference['confidence_source'] != 'raw'
            or amplitude['status'] != 'fused' or amplitude['confidence_source'] != 'raw'
            or not amplitude['provenance_verified'] or amplitude['inference'] != inference):
        raise ValueError('Incomplete or incompatible paired inference/fusion')
    for relative, expected in inference['files_sha256'].items():
        if sha256(paired / relative) != expected:
            raise ValueError('Saved inference file changed: ' + relative)
    fusion_rows = []
    for alpha, source_depth in (('0.0', 'depth_est_raw'), ('1.0', 'depth_est')):
        group = amplitude['groups'][alpha]
        fusion_dir = Path(group['fusion_dir'])
        fusion = json.loads((fusion_dir / 'fusion_manifest.json').read_text())
        if fusion['status'] != 'complete' or fusion['scans'] != amplitude['scans']:
            raise ValueError('Incomplete fusion: ' + str(fusion_dir))
        for name in ('prob_threshold', 'disp_threshold', 'num_consistent'):
            if fusion[name] != amplitude['protocol'][name]:
                raise ValueError('Fusion protocol mismatch: ' + name)
        counts = {}
        for scan in amplitude['scans']:
            for folder in ('depth_est', 'confidence', 'cams', 'images'):
                original_folder = source_depth if folder == 'depth_est' else folder
                originals = sorted((paired / scan / original_folder).iterdir())
                copies = sorted(p for p in (Path(group['path']) / scan / folder).iterdir()
                                if not (folder == 'depth_est' and p.name.endswith('_prob_filtered.pfm')))
                if [p.name for p in originals] != [p.name for p in copies]:
                    raise ValueError('Fusion input file set mismatch')
                for original, copied in zip(originals, copies):
                    expected = inference['files_sha256'][str(original.relative_to(paired))]
                    if sha256(copied) != expected:
                        raise ValueError('Fusion endpoint or shared input mismatch: ' + str(copied))
                    if folder == 'depth_est':
                        depth = read_pfm(str(copied))[0]
                        confidence = read_pfm(str(Path(group['path']) / scan / 'confidence' / copied.name))[0]
                        if not np.isfinite(depth).all() or not np.isfinite(confidence).all():
                            raise ValueError('Nonfinite saved depth/confidence')
                        depth[confidence < fusion['prob_threshold']] = 0
                        filtered = read_pfm(str(copied.with_name(copied.stem + '_prob_filtered.pfm')))[0]
                        if not np.array_equal(depth, filtered):
                            raise ValueError('Photometric filtering mismatch: ' + str(copied))
            artifact = fusion['outputs'][scan]
            if sha256(artifact['path']) != artifact['sha256']:
                raise ValueError('Fused PLY changed: ' + artifact['path'])
            vertices = PlyData.read(artifact['path'])['vertex']
            if vertices.count < 1 or any(not np.isfinite(vertices[axis]).all() for axis in ('x', 'y', 'z')):
                raise ValueError('Empty/nonfinite point cloud: ' + artifact['path'])
            counts[scan] = vertices.count
        fusion_rows.append({'depth_source': 'raw' if alpha == '0.0' else 'refined',
                            'confidence_source': 'raw', 'fusion_dir': str(fusion_dir),
                            'point_counts': counts, 'total': sum(counts.values()),
                            'photometric_candidate_rates': {
                                scan: group['scenes'][scan]['photometric_candidate_rate']
                                for scan in amplitude['scans']}})
    report = {'status': 'verified', 'validation_summary_sha256': sha256(validation_path),
              'inference_manifest_sha256': sha256(paired / 'inference_manifest.json'),
              'script_sha256': sha256(__file__), 'validation_rows': validation_rows,
              'protected_validation_outputs_exact': True,
              'inference_file_count': len(inference['files_sha256']),
              'fusion_endpoint_and_shared_inputs_exact': True,
              'checkpoint_sha256': inference['checkpoint_sha256'],
              'protocol': amplitude['protocol'], 'fusion_rows': fusion_rows,
              'dtu_metrics': 'not evaluated; point counts are not accuracy/completeness scores',
              'missing_statistics': amplitude['missing_statistics']}
    write_json(output_path, report)
    print(json.dumps({'status': report['status'], 'fusion_rows': fusion_rows}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation_summary', required=True)
    parser.add_argument('--test_root', required=True)
    parser.add_argument('--output_json', required=True)
    args = parser.parse_args()
    verify(args.validation_summary, args.test_root, args.output_json)
