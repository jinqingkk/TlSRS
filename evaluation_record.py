"""Provenance for paired SGER evaluations (Python 3.7 compatible)."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def source_record():
    root = Path(__file__).resolve().parent
    names = ('test.py', 'train.py', 'gipuma.py', 'evaluation_record.py',
             'models/cas_mvsnet.py', 'models/module.py',
             'models/sger_refinement.py', 'datasets/general_eval.py', 'datasets/dtu_yao.py',
             'gradient_audit.py', 'gradient_audit_utils.py',
             'experiment15_d_probe.py', 'experiment15_d_diagnose.py')
    def git(*args):
        result = subprocess.run(['git'] + list(args), cwd=str(root),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode('utf-8', errors='replace').strip() if result.returncode == 0 else None
    return {'sha256': {name: sha256(root / name) for name in names},
            'git_head': git('rev-parse', 'HEAD'), 'git_status': git('status', '--short')}


def start_inference(args, scans):
    out = Path(args.outdir)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Inference requires an empty output directory: ' + str(out))
    checkpoint_hash = sha256(args.loadckpt)
    out.mkdir(parents=True, exist_ok=True)
    record = {'status': 'running', 'started_at': datetime.now().isoformat(),
              'command': sys.argv, 'arguments': vars(args).copy(), 'scans': scans,
              'checkpoint': str(Path(args.loadckpt).resolve()),
              'checkpoint_sha256': checkpoint_hash, 'source': source_record(),
              'depth_sources': {'depth_est': 'refined', 'depth_est_raw': 'raw'},
              'confidence_source': 'raw', 'paired_forward': True}
    write_json(out / 'inference_manifest.json', record)
    return record
