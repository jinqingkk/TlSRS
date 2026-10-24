"""Paired raw/refined reprojection diagnostics; NOT Gipuma acceptance masks."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from datasets.data_io import read_pfm
from evaluation_record import sha256, write_json
from gipuma import read_camera_parameters


def reprojection(depth_ref, camera_ref, depth_src, camera_src):
    """Return support, pixel round-trip error and relative depth error.

    Support requires positive camera-space Z, in-bounds interpolation and four
    valid source neighbours. It is not an occlusion ground-truth label.
    """
    kr, er = camera_ref
    ks, es = camera_src
    height, width = depth_ref.shape
    y, x = np.indices((height, width), dtype=np.float32)
    pixels = np.stack((x.ravel(), y.ravel(), np.ones(depth_ref.size, np.float32)))
    xyz = np.linalg.inv(kr).dot(pixels) * depth_ref.ravel()
    transform = es.dot(np.linalg.inv(er))
    xyz_src = transform[:3, :3].dot(xyz) + transform[:3, 3:4]
    projected = ks.dot(xyz_src)
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = projected[:2] / projected[2:3]
    sx, sy = uv.reshape(2, height, width)
    support = (np.isfinite(depth_ref) & (depth_ref > 0) & np.isfinite(sx) & np.isfinite(sy)
               & (xyz_src[2].reshape(height, width) > 0)
               & (sx >= 0) & (sy >= 0) & (sx < depth_src.shape[1]-1) & (sy < depth_src.shape[0]-1))
    sx = np.where(support, sx, -1).astype(np.float32)
    sy = np.where(support, sy, -1).astype(np.float32)
    valid_src = np.isfinite(depth_src) & (depth_src > 0)
    sampled = cv2.remap(np.where(valid_src, depth_src, 0).astype(np.float32), sx, sy, cv2.INTER_LINEAR)
    coverage = cv2.remap(valid_src.astype(np.float32), sx, sy, cv2.INTER_LINEAR)
    support &= (coverage >= 1-1e-6) & (sampled > 0)
    source_pixels = np.stack((sx.ravel(), sy.ravel(), np.ones(depth_ref.size, np.float32)))
    source_xyz = np.linalg.inv(ks).dot(source_pixels) * sampled.ravel()
    inverse = np.linalg.inv(transform)
    back = inverse[:3, :3].dot(source_xyz) + inverse[:3, 3:4]
    back_pixels = kr.dot(back)
    with np.errstate(divide='ignore', invalid='ignore'):
        uv_back = (back_pixels[:2] / back_pixels[2:3]).reshape(2, height, width)
        distance = np.sqrt((uv_back[0]-x)**2 + (uv_back[1]-y)**2)
        relative = np.abs(back[2].reshape(height, width)-depth_ref) / depth_ref
    support &= (back[2].reshape(height, width) > 0) & np.isfinite(distance) & np.isfinite(relative)
    return support, distance, relative


def read_pairs(path):
    result = {}
    with open(str(path)) as stream:
        for _ in range(int(stream.readline())):
            ref = int(stream.readline())
            result[ref] = [int(v) for v in stream.readline().split()[1::2]]
    return result


def run(args):
    source, out = Path(args.source_dir), Path(args.output_dir)
    inference_path = source/'inference_manifest.json'
    inference = json.loads(inference_path.read_text())
    if inference['status'] != 'complete' or not inference['paired_forward']:
        raise ValueError('Use completed paired inference')
    if args.sources < 1 or not 1 <= args.min_consistent <= args.sources:
        raise ValueError('Invalid source-view count')
    scans = Path(args.scan_list).read_text().split()
    if not set(scans).issubset(inference['scans']):
        raise ValueError('Scene list differs from saved inference')
    out.mkdir(parents=True, exist_ok=False)
    records, hashes = [], {}
    manifest = {'command': sys.argv, 'arguments': vars(args), 'inference_manifest_sha256': sha256(inference_path),
                'checkpoint_sha256': inference['checkpoint_sha256'], 'checkpoint_epoch': inference['checkpoint_epoch'],
                'script_sha256': sha256(__file__), 'status': 'running',
                'protocol': 'diagnostic proxy: <1 pixel and <0.01 relative depth; NOT Gipuma disp_threshold',
                'aggregation': 'pixel pooled; same raw confidence >=0.9; pair means on intersection of supports',
                'occlusion': 'not measured; insufficient support must not be called confirmed occlusion',
                'inputs_sha256': hashes, 'records': records}
    write_json(out/'summary.json', manifest)
    def verified(path):
        key = str(path.relative_to(source))
        digest = sha256(path)
        if inference['files_sha256'].get(key) != digest:
            raise ValueError('Input changed since inference: ' + key)
        hashes[key] = digest
        return str(path)
    for scan in scans:
        pair_path = Path(args.testpath)/scan/'pair.txt'
        hashes[str(pair_path)] = sha256(pair_path)
        pairs = read_pairs(pair_path)
        for ref in [int(v) for v in args.views.split(',')]:
            views = [ref] + pairs[ref][:args.sources]
            cameras = {v: read_camera_parameters(verified(source/scan/'cams'/('%08d_cam.txt' % v))) for v in views}
            depths = {mode: {v: read_pfm(verified(source/scan/folder/('%08d.pfm' % v)))[0] for v in views}
                      for mode, folder in (('raw', 'depth_est_raw'), ('refined', 'depth_est'))}
            confidence = read_pfm(verified(source/scan/'confidence'/('%08d.pfm' % ref)))[0]
            common_ref = np.isfinite(confidence) & (confidence >= .9)
            for mode in depths:
                common_ref &= np.isfinite(depths[mode][ref]) & (depths[mode][ref] > 0)
            counts = {mode: np.zeros(confidence.shape, np.int32) for mode in depths}
            supports = {mode: np.zeros(confidence.shape, np.int32) for mode in depths}
            row = {'scan': scan, 'reference': ref, 'sources': views[1:],
                   'resolution': list(confidence.shape), 'candidate_count': int(common_ref.sum()), 'pairs': []}
            for src in views[1:]:
                outputs = {mode: reprojection(depths[mode][ref], cameras[ref], depths[mode][src], cameras[src])
                           for mode in depths}
                shared = common_ref & outputs['raw'][0] & outputs['refined'][0]
                pair = {'source': src, 'common_support_pixels': int(shared.sum())}
                for mode, (support, distance, relative) in outputs.items():
                    supports[mode] += support.astype(np.int32)
                    counts[mode] += (support & (distance < 1) & (relative < .01)).astype(np.int32)
                    pair[mode] = {'pixel_error_sum': float(distance[shared].sum(dtype=np.float64)),
                                  'relative_depth_error_sum': float(relative[shared].sum(dtype=np.float64))}
                row['pairs'].append(pair)
            for mode in depths:
                row[mode] = {'consistent_view_histogram': np.bincount(counts[mode][common_ref], minlength=args.sources+1).tolist(),
                             'support_view_histogram': np.bincount(supports[mode][common_ref], minlength=args.sources+1).tolist(),
                             'geometric_candidate_pass_count': int((common_ref & (counts[mode] >= args.min_consistent)).sum())}
            records.append(row)
            write_json(out/'summary.json', manifest)
            print(scan, ref, {m: row[m]['geometric_candidate_pass_count'] for m in depths}, flush=True)
    manifest['status'] = 'complete'
    write_json(out/'summary.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_dir', required=True)
    parser.add_argument('--testpath', required=True)
    parser.add_argument('--scan_list', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--views', default='0,24,48')
    parser.add_argument('--sources', type=int, default=5)
    parser.add_argument('--min_consistent', type=int, default=4)
    run(parser.parse_args())
