"""Evaluate joint Det/OCC PTH or ONNX; no rendering or video generation."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bev_occ_inference import (add_inference_arguments, checked_occupancy, load_context,
                               make_loader, make_predictor, unpack_batch)
from bev_occ_reporting import create_report_run, file_identity, save_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inference_arguments(parser)
    parser.add_argument('--out-dir', default='work_dirs/bev_occ_eval')
    parser.add_argument('--limit', '--smoke-test', dest='limit', type=int, default=0,
                        help='First N samples; 0=full set. Partial runs skip official detection metrics.')
    parser.add_argument('--skip-det-eval', action='store_true', help='Save detections and OCC metrics; skip nuScenes detection metrics')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    import mmcv
    import numpy as np
    from mmdet3d.datasets.nuscenes_det_occ_dataset import OCC_CLASSES
    cfg, dataset = load_context(args)
    indices = list(range(min(args.limit, len(dataset)) if args.limit else len(dataset)))
    if not indices:
        parser.error('Dataset is empty')
    partial = len(indices) != len(dataset)
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive, metadata = create_report_run(output, dict(
        config=file_identity(args.config), checkpoint=file_identity(args.model),
        backend=args.backend, command=sys.argv, arguments=vars(args), num_samples=len(indices),
        dataset_samples=len(dataset), sample_indices=indices, partial=partial, status='running'),
        resolved_config=cfg.pretty_text)
    print('Evaluation archive:', archive)
    hist = np.zeros((18, 18), dtype=np.int64)
    detections = []
    predictor = None
    progress = mmcv.ProgressBar(len(indices))
    try:
        for index, batch in zip(indices, make_loader(dataset, indices, args.workers)):
            gt, mask = dataset.load_occ_ground_truth(index)
            data = unpack_batch(batch, args)
            if predictor is None:
                predictor = make_predictor(args, cfg, dataset, data)
                metadata['note'] = predictor.description
                print(predictor.description)
            result = predictor(data)
            pred = checked_occupancy(result, gt)
            hist += np.bincount(18 * gt[mask].astype(np.int64) + pred[mask].astype(np.int64), minlength=324).reshape(18, 18)
            detections.append({'pts_bbox': result['pts_bbox']})
            progress.update()
        mmcv.dump(detections, str(output / 'detections.pkl'))
        np.save(output / 'occ_confusion.npy', hist)
        union = hist.sum(0) + hist.sum(1) - np.diag(hist)
        iou = np.divide(np.diag(hist), union, out=np.full(18, np.nan), where=union > 0) * 100
        metrics = {'occ/' + name + '_IoU': float(iou[i]) for i, name in enumerate(OCC_CLASSES)}
        finite = iou[:17][np.isfinite(iou[:17])]
        metrics['occ/mIoU'] = float(finite.mean()) if finite.size else float('nan')
        (output / 'occ_metrics.json').write_text(json.dumps(metrics, indent=2))
        status = 'partial_occ_complete_detection_skipped' if partial else 'occ_complete_detection_skipped'
        if not (partial or args.skip_det_eval):
            save_report(output, archive, metrics, metadata, status='occ_complete_detection_pending')
            det_metrics = dataset.evaluate(detections, metric='bbox', jsonfile_prefix=str(output / 'detection'))
            metrics.update({k: float(v) for k, v in det_metrics.items()})
            status = 'complete'
        save_report(output, archive, metrics, metadata, status=status)
    except Exception as exc:
        # Preserve completed OCC metrics if the detection evaluator fails.
        metadata['error'] = str(exc)
        save_report(output, archive, locals().get('metrics', {}), metadata, status='failed')
        raise
    if partial:
        print('\nPARTIAL evaluation: official detection metrics skipped; not a full validation result.')
    print('\n' + (output / 'metrics.txt').read_text())
    print('Results:', output)


if __name__ == '__main__':
    main()
