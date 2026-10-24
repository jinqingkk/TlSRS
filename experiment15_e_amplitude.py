"""Prepare/fuse the preregistered alpha=0,.25,.5,1 paired depth diagnostic."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from datasets.data_io import read_pfm, save_pfm
from evaluation_record import sha256, source_record, write_json
from gipuma import gipuma_filter

ALPHAS = (0.0, 0.25, 0.5, 1.0)


def interpolate(raw, refined, alpha):
    if raw.shape != refined.shape or raw.ndim != 2:
        raise ValueError('Raw/refined depth shapes differ')
    if not np.isfinite(raw).all() or not np.isfinite(refined).all():
        raise ValueError('Nonfinite saved depth')
    # Preserve exact endpoint maps, including their float32 rounding.
    if alpha == 0:
        return raw.copy()
    if alpha == 1:
        return refined.copy()
    return (raw + alpha * (refined - raw)).astype(np.float32)


def run(args):
    source, out = Path(args.source_dir), Path(args.output_dir)
    alphas = tuple(float(v) for v in getattr(args, 'alphas', '0,0.25,0.5,1').split(','))
    if not alphas or len(set(alphas)) != len(alphas) or any(a not in ALPHAS for a in alphas):
        raise ValueError('Select distinct preregistered amplitudes from 0,0.25,0.5,1')
    scans = Path(args.scan_list).read_text().split()
    if not scans or len(scans) != len(set(scans)):
        raise ValueError('Scene list must be nonempty and unique')
    manifest_path = source / 'inference_manifest.json'
    provenance = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if provenance is None:
        if not args.allow_unverified_source:
            raise ValueError('Missing inference manifest; regenerate paired maps or explicitly label legacy data unverified')
    elif (provenance['status'] != 'complete' or not provenance['paired_forward']
          or not provenance.get('files_sha256')
          or not set(scans).issubset(provenance['scans'])):
        raise ValueError('Incomplete or incompatible inference provenance')
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'preparing', 'command': sys.argv, 'arguments': vars(args),
                'source': source_record(), 'script_sha256': sha256(__file__),
                'source_dir': str(source.resolve()), 'inference': provenance,
                'provenance_verified': provenance is not None,
                'scan_list_sha256': sha256(args.scan_list), 'scans': scans,
                'alphas': alphas, 'confidence_source': 'raw',
                'protocol': {'filter': 'gipuma', 'prob_threshold': args.prob_threshold,
                             'disp_threshold': args.disp_threshold, 'num_consistent': args.num_consistent},
                'inputs_sha256': {}, 'groups': {},
                'missing_statistics': ['Gipuma geometric/final mask rates', 'consistent-view histogram'],
                'interpretation': 'Amplitude diagnostic only; alpha=0 is a control, not a final SGER architecture'}
    write_json(out/'amplitude_manifest.json', manifest)
    for alpha in alphas:
        group = out / ('alpha_%g' % alpha)
        manifest['groups'][str(alpha)] = {'path': str(group), 'scenes': {}}
        for scan in scans:
            original, destination = source/scan, group/scan
            destination.mkdir(parents=True)
            for folder in ('cams', 'images', 'confidence'):
                shutil.copytree(str(original/folder), str(destination/folder))
            depth_dir = destination/'depth_est'
            depth_dir.mkdir()
            images = sorted((original/'images').glob('*.jpg'))
            if not images:
                raise ValueError('No reference images: ' + str(original))
            photo_count, pixels = 0, 0
            for image in images:
                name = image.stem
                raw_path = original/'depth_est_raw'/(name+'.pfm')
                refined_path = original/'depth_est'/(name+'.pfm')
                confidence_path = original/'confidence'/(name+'.pfm')
                camera = original/'cams'/(name+'_cam.txt')
                for path in (raw_path, refined_path, confidence_path, camera, image):
                    key = str(path.relative_to(source))
                    digest = sha256(path)
                    if key in manifest['inputs_sha256'] and manifest['inputs_sha256'][key] != digest:
                        raise RuntimeError('Source changed during amplitude preparation: ' + key)
                    manifest['inputs_sha256'][key] = digest
                    if provenance and 'files_sha256' in provenance and provenance['files_sha256'].get(key) != digest:
                        raise RuntimeError('Source differs from inference manifest: ' + key)
                raw, refined = read_pfm(str(raw_path))[0], read_pfm(str(refined_path))[0]
                confidence = read_pfm(str(confidence_path))[0]
                if confidence.shape != raw.shape or not np.isfinite(confidence).all():
                    raise ValueError('Invalid raw confidence')
                depth = interpolate(raw, refined, alpha)
                save_pfm(str(depth_dir/(name+'.pfm')), depth)
                photo_count += int(((confidence >= args.prob_threshold) & (depth > 0)).sum())
                pixels += depth.size
            manifest['groups'][str(alpha)]['scenes'][scan] = {
                'views': len(images), 'pixels': pixels, 'photometric_candidate_pixels': photo_count,
                'photometric_candidate_rate': photo_count/pixels}
        print('Prepared alpha', alpha, flush=True)
        write_json(out/'amplitude_manifest.json', manifest)
    manifest['status'] = 'prepared'
    write_json(out/'amplitude_manifest.json', manifest)
    if args.fuse:
        for alpha in alphas:
            group = manifest['groups'][str(alpha)]
            group['fusion_dir'] = gipuma_filter(scans, group['path'], args.prob_threshold,
                                               args.disp_threshold, args.num_consistent,
                                               args.fusibile_exe_path)
            write_json(out/'amplitude_manifest.json', manifest)
        manifest['status'] = 'fused'
        write_json(out/'amplitude_manifest.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--scan_list', required=True)
    parser.add_argument('--alphas', default='0,0.25,0.5,1', help='fixed diagnostic groups; use 0,1 for raw/refined comparison')
    parser.add_argument('--allow_unverified_source', action='store_true')
    parser.add_argument('--fuse', action='store_true')
    parser.add_argument('--prob_threshold', type=float, default=.9)
    parser.add_argument('--disp_threshold', type=float, default=.25)
    parser.add_argument('--num_consistent', type=int, default=4)
    parser.add_argument('--fusibile_exe_path', default='../../fusibile-master/build/fusibile')
    run(parser.parse_args())
