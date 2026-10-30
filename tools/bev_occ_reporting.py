"""Human-readable evaluation reports and immutable per-run result directories."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

DET_PREFIX = 'pts_bbox_NuScenes/'


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def create_report_run(output, metadata, resolved_config=None):
    output = Path(output)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid4().hex[:8]
    archive = output / 'reports' / run_id
    archive.mkdir(parents=True, exist_ok=False)
    metadata = dict(metadata, run_id=run_id,
                    recorded_at_utc=datetime.now(timezone.utc).isoformat())
    if resolved_config is not None:
        (archive / 'config.py').write_text(resolved_config, encoding='utf-8')
    (archive / 'run_metadata.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    return archive, metadata


def number(value):
    if value is None:
        return 'N/A'
    return f'{value:.4f}' if math.isfinite(value) else 'nan'


def format_report(metrics, metadata):
    rows = ['BEVDet + FlashOCC evaluation',
            'Status: ' + metadata.get('status', 'unknown'),
            'Run: ' + metadata.get('run_id', 'unknown'),
            'Recorded UTC: ' + metadata.get('recorded_at_utc', 'unknown')]
    if metadata.get('backend'):
        rows.append('Backend: ' + metadata['backend'])
    if metadata.get('partial'):
        rows.append('Scope: PARTIAL dataset; not an official full validation result')
    for name in ('config', 'checkpoint'):
        identity = metadata.get(name) or {}
        rows.append(name + ': ' + identity.get('path', 'not recorded'))
        if identity.get('sha256'):
            rows.append(name + ' SHA256: ' + identity['sha256'])
    if metadata.get('note'):
        rows.append('Note: ' + metadata['note'])
    if 'num_samples' in metadata:
        rows.append('Evaluated samples: ' + str(metadata['num_samples']))
    rows.extend(['', 'Detection metrics (0-1 scores; errors in official units):'])
    for name in ('mAP', 'NDS', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE'):
        rows.append(f'{name}: {number(metrics.get(DET_PREFIX + name))}')
    rows.extend(['', 'Detection per class (AP = mean over 0.5/1/2/4 m thresholds):',
                 'Class\tAP\tATE\tASE\tAOE\tAVE\tAAE'])
    classes = [k[len(DET_PREFIX):-len('_AP_dist_0.5')]
               for k in metrics if k.startswith(DET_PREFIX) and k.endswith('_AP_dist_0.5')]
    for cls in classes:
        ap = [metrics.get(DET_PREFIX + cls + '_AP_dist_' + distance)
              for distance in ('0.5', '1.0', '2.0', '4.0')]
        average = sum(ap) / 4 if all(v is not None for v in ap) else None
        errors = [metrics.get(DET_PREFIX + cls + '_' + suffix)
                  for suffix in ('trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err')]
        rows.append('\t'.join([cls] + [number(v) for v in [average] + errors]))
    rows.extend(['', 'Occupancy IoU (%; mIoU excludes free):',
                 'mIoU: ' + number(metrics.get('occ/mIoU'))])
    rows.extend(k[4:] + ': ' + number(v) for k, v in metrics.items()
                if k.startswith('occ/') and k != 'occ/mIoU')
    return '\n'.join(rows) + '\n'


def save_report(output, archive, metrics, metadata, status):
    output, archive = Path(output), Path(archive)
    metadata = dict(metadata, status=status)
    files = {
        'metrics.json': json.dumps(metrics, indent=2, ensure_ascii=False),
        'metrics.txt': format_report(metrics, metadata),
        'run_metadata.json': json.dumps(metadata, indent=2, ensure_ascii=False),
    }
    for directory in (archive, output):
        for name, contents in files.items():
            (directory / name).write_text(contents, encoding='utf-8')
