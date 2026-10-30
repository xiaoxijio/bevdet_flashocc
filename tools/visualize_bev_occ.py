"""Render joint Det/OCC PTH or ONNX into scene images/videos; no metric evaluation."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bev_occ_inference import (add_inference_arguments, checked_occupancy, load_context,
                               make_loader, make_predictor, unpack_batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inference_arguments(parser)
    parser.add_argument('--out-dir', default='work_dirs/bev_occ_visualizations')
    parser.add_argument('--scene', nargs='+', help='Scene names or tokens')
    parser.add_argument('--scene-count', type=int, default=1, help='First N scenes; 0=all')
    parser.add_argument('--vis-count', '--limit', dest='vis_count', type=int, default=0, help='Max frames; 0=all selected frames')
    parser.add_argument('--vis-every', type=int, default=1)
    parser.add_argument('--score-thr', type=float, default=.3)
    parser.add_argument('--fps', type=float, default=2.)
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--save-occupancy', action='store_true')
    parser.add_argument('--occ-view', choices=('3d', 'bev'), default='3d')
    parser.add_argument('--occ-mask', choices=('camera', 'all'), default='camera',
                        help='Use the same camera visibility mask for prediction and GT, or show all voxels')
    args = parser.parse_args()
    if args.vis_every < 1 or min(args.vis_count, args.scene_count) < 0 or args.fps <= 0 or not 0 <= args.score_thr <= 1:
        parser.error('Invalid frame count, stride, FPS or threshold')
    import mmcv
    import numpy as np
    from bev_occ_visualization import SceneRenderer, select_frames, scene_name
    from bev_occ_reporting import file_identity
    cfg, dataset = load_context(args)
    indices = select_frames(dataset.data_infos, args.scene, args.scene_count, args.vis_every, args.vis_count)
    if not indices:
        parser.error('No frames selected')
    renderer = SceneRenderer(args.out_dir, threshold=args.score_thr, fps=args.fps,
                             video=not args.no_video, save_occupancy=args.save_occupancy,
                             occ_view=args.occ_view, backend=args.backend, mask_mode=args.occ_mask)
    renderer.expected_counts = Counter(scene_name(dataset.data_infos[i]) for i in indices)
    predictor = None
    progress = mmcv.ProgressBar(len(indices))
    try:
        for index, batch in zip(indices, make_loader(dataset, indices, args.workers)):
            gt, camera_mask = dataset.load_occ_ground_truth(index)
            data = unpack_batch(batch, args)
            if predictor is None:
                predictor = make_predictor(args, cfg, dataset, data)
                print(predictor.description)
            result = predictor(data)
            checked_occupancy(result, gt)
            mask = camera_mask if args.occ_mask == 'camera' else np.ones_like(camera_mask, dtype=bool)
            renderer.save(dataset, index, result, gt, mask)
            progress.update()
    finally:
        renderer.close()
    metadata = dict(backend=args.backend, model=file_identity(args.model), config=file_identity(args.config),
                    arguments=vars(args), inference=predictor.description,
                    num_frames=len(indices), scene_frames=dict(renderer.counts))
    (Path(args.out_dir) / 'visualization_metadata.json').write_text(json.dumps(metadata, indent=2))
    print('\nVisualizations:', Path(args.out_dir).resolve())
    print('Scene frames:', dict(renderer.counts))


if __name__ == '__main__':
    main()
