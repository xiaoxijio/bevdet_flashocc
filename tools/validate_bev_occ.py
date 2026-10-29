"""Evaluate joint Det/OCC checkpoint and save headless prediction/GT visualizations."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.datasets.nuscenes_det_occ_dataset import OCC_CLASSES
from mmdet3d.models import build_model
from pyquaternion import Quaternion

from bev_occ_visualization import SceneRenderer, select_frames, scene_name
from bev_occ_reporting import create_report_run, file_identity, save_report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config');p.add_argument('checkpoint')
    p.add_argument('--out-dir',default='work_dirs/eval_epoch20')
    p.add_argument('--vis-count',type=int,default=0,help='Max rendered frames; 0 = all selected scene frames')
    p.add_argument('--vis-every',type=int,default=1)
    p.add_argument('--score-thr',type=float,default=.3)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--gpu-id',type=int,default=0)
    p.add_argument('--smoke-test',type=int,default=0,help='Only run N selected frames; no official evaluation')
    p.add_argument('--vis-only',action='store_true',help='Infer selected scenes only; no metrics/evaluation')
    p.add_argument('--scene',nargs='+',help='Scene name(s), e.g. scene-0268, or scene tokens')
    p.add_argument('--scene-count',type=int,default=1,help='First N chronological scenes; 0 = all')
    p.add_argument('--fps',type=float,default=2.,help='Video frames/sec; nuScenes keyframes are about 2 Hz')
    p.add_argument('--no-video',action='store_true',help='Save combined JPEG frames without MP4')
    p.add_argument('--save-occupancy',action='store_true',help='Also save per-frame NPZ, in the same flat directory')
    p.add_argument('--skip-det-eval',action='store_true',help='Save detections but skip memory-heavy nuScenes bbox metrics')
    args=p.parse_args()
    if args.vis_every<1 or args.vis_count<0 or args.smoke_test<0 or args.scene_count<0 or args.fps<=0: p.error('Invalid sample counts/fps')
    cfg=Config.fromfile(args.config)
    cfg.data.test.test_mode=True
    cfg.model.train_cfg=None;cfg.model.img_backbone.pretrained=None
    dataset=build_dataset(cfg.data.test)
    selected=select_frames(dataset.data_infos,args.scene,args.scene_count,args.vis_every,args.vis_count)
    if not selected: p.error('No frames selected')
    run_indices=selected if (args.vis_only or args.smoke_test) else list(range(len(dataset)))
    if args.smoke_test: run_indices=run_indices[:args.smoke_test]
    selected_set=set(selected) & set(run_indices)
    # Fail early on labels only for frames that will actually be processed.
    for i in run_indices:
        if not Path(dataset._occ_path(i)).is_file(): raise FileNotFoundError(dataset._occ_path(i))
    output=Path(args.out_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    report_archive = report_metadata = None
    if not (args.vis_only or args.smoke_test):
        report_archive, report_metadata = create_report_run(output, dict(
            config=file_identity(args.config), checkpoint=file_identity(args.checkpoint),
            command=sys.argv, arguments=vars(args), num_samples=len(run_indices),
            status='running'), resolved_config=cfg.pretty_text)
        print('Evaluation archive:', report_archive)
    model=build_model(cfg.model,test_cfg=cfg.get('test_cfg'))
    checkpoint=torch.load(args.checkpoint,map_location='cpu')
    state=checkpoint.get('state_dict',checkpoint)
    model.load_state_dict({k[7:] if k.startswith('module.') else k:v for k,v in state.items()},strict=True)
    model.CLASSES=dataset.CLASSES
    model=MMDataParallel(model.cuda(args.gpu_id),device_ids=[args.gpu_id]);model.eval()
    subset=torch.utils.data.Subset(dataset,run_indices)
    loader=build_dataloader(subset,samples_per_gpu=1,workers_per_gpu=args.workers,dist=False,shuffle=False)
    hist=np.zeros((18,18),dtype=np.int64);detections=[]
    renderer=SceneRenderer(output/'visualizations',threshold=args.score_thr,
                           fps=args.fps,video=not args.no_video,
                           save_occupancy=args.save_occupancy)
    from collections import Counter
    renderer.expected_counts=Counter(scene_name(dataset.data_infos[i]) for i in selected_set)
    progress=mmcv.ProgressBar(len(run_indices))
    try:
        for position,data in enumerate(loader):
            index=run_indices[position]
            with torch.no_grad(): result=model(return_loss=False,rescale=True,**data)[0]
            pred=np.asarray(result['occ'])
            gt,mask=dataset.load_occ_ground_truth(index)
            if pred.shape!=gt.shape:
                raise ValueError(f'Occupancy shape mismatch: prediction={pred.shape}, GT={gt.shape}, occ_crop={dataset.occ_crop}')
            if np.any((pred<0)|(pred>=18)):
                raise ValueError('Invalid occupancy class IDs; expected labels 0..17')
            if not (args.vis_only or args.smoke_test):
                hist+=np.bincount(18*gt[mask].astype(np.int64)+pred[mask].astype(np.int64),minlength=324).reshape(18,18)
                detections.append({'pts_bbox':result['pts_bbox']})
            if index in selected_set:
                renderer.save(dataset,index,result,gt,mask)
            progress.update()
    finally:
        renderer.close()
    if args.vis_only or args.smoke_test:
        print('\nVisualization complete; metrics skipped:',output/'visualizations')
        print('Scene frames:',dict(renderer.counts))
        return
    # Persist detection outputs first; permit bbox re-evaluation without inference.
    mmcv.dump(detections,str(output/'detections.pkl'))
    union=hist.sum(0)+hist.sum(1)-np.diag(hist)
    iou=np.divide(np.diag(hist),union,out=np.full(18,np.nan),where=union>0)*100
    metrics={'occ/'+name+'_IoU':float(iou[i]) for i,name in enumerate(OCC_CLASSES)}
    finite=iou[:17][np.isfinite(iou[:17])]
    metrics['occ/mIoU']=float(finite.mean()) if finite.size else float('nan')
    # Same mask and mIoU definition as NuScenesDatasetDetOcc._evaluate_occ.
    np.save(output/'occ_confusion.npy',hist)
    (output/'occ_metrics.json').write_text(json.dumps(metrics,indent=2))
    print('\nOccupancy metrics:',json.dumps(metrics,indent=2))
    # Persist OCC report before the potentially memory-heavy detection evaluator.
    save_report(output, report_archive, metrics, report_metadata,
                status='occ_complete_detection_skipped' if args.skip_det_eval
                else 'occ_complete_detection_pending')
    if not args.skip_det_eval:
        det_metrics=dataset.evaluate(detections,metric='bbox',jsonfile_prefix=str(output/'detection'))
        metrics.update({k:float(v) for k,v in det_metrics.items()})
    save_report(output, report_archive, metrics, report_metadata,
                status='occ_complete_detection_skipped' if args.skip_det_eval else 'complete')
    print('Text report:', output/'metrics.txt')
    print('Archived results:', report_archive)
    print('All metrics:',json.dumps(metrics,indent=2));print('Results:',output)

if __name__=='__main__':main()
