"""Reports must retain prior runs, units and incomplete-evaluation status."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from bev_occ_reporting import create_report_run, file_identity, save_report


def test_reports_keep_history_and_mark_partial_metrics(tmp_path):
    weights = tmp_path / 'weights.pth'
    weights.write_bytes(b'checkpoint identity')
    identity = file_identity(weights)
    assert len(identity['sha256']) == 64
    assert identity['path'] == str(weights.resolve())
    first, metadata = create_report_run(tmp_path, dict(checkpoint=identity),
                                        resolved_config='model = dict()\n')
    partial = {'occ/mIoU': 29.967998437525967}
    save_report(tmp_path, first, partial, metadata, 'occ_complete_detection_pending')
    assert 'mAP: N/A' in (first / 'metrics.txt').read_text()
    assert 'mIoU: 29.9680' in (first / 'metrics.txt').read_text()
    full = dict(partial, **{'pts_bbox_NuScenes/mAP': .29264895505602884,
                           'pts_bbox_NuScenes/NDS': .3955572881642055})
    for distance in ('0.5', '1.0', '2.0', '4.0'):
        full['pts_bbox_NuScenes/car_AP_dist_' + distance] = .5
    full['pts_bbox_NuScenes/car_vel_err'] = float('nan')
    save_report(tmp_path, first, full, metadata, 'complete')
    text = (first / 'metrics.txt').read_text()
    assert 'mAP: 0.2926' in text and 'NDS: 0.3956' in text
    assert 'car\t0.5000\tN/A\tN/A\tN/A\tnan\tN/A' in text
    second, next_metadata = create_report_run(tmp_path, {})
    save_report(tmp_path, second, partial, next_metadata, 'occ_complete_detection_skipped')
    assert first != second
    assert json.loads((first / 'metrics.json').read_text())['pts_bbox_NuScenes/mAP'] == full['pts_bbox_NuScenes/mAP']
    assert 'pts_bbox_NuScenes/mAP' not in json.loads((tmp_path / 'metrics.json').read_text())
    assert (first / 'config.py').read_text() == 'model = dict()\n'
