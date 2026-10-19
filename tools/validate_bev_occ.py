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

PALETTE = np.array([[0,0,0],[255,120,50],[255,192,203],[255,255,0],
 [0,150,245],[0,255,255],[200,180,0],[255,0,0],[255,240,150],
 [135,60,0],[160,32,240],[255,0,255],[175,0,75],[75,0,75],
 [150,240,80],[230,230,250],[0,175,0],[245,245,245]], dtype=np.uint8)
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def transform(rotation, translation):
    t = np.eye(4)
    t[:3,:3] = Quaternion(rotation).rotation_matrix
    t[:3,3] = translation
    return t

def occ_top(labels, mask):
    valid = mask & (labels < 17)
    z = np.arange(labels.shape[2])[None,None,:]
    highest = np.where(valid,z,-1).max(axis=2)
    top = np.take_along_axis(labels,np.maximum(highest,0)[...,None],axis=2)[...,0]
    top = np.where(highest>=0,top,17)
    # Screen: forward +x is up; ego left +y is left.
    return PALETTE[top[::-1,::-1]]

def save_vis(dataset, index, result, gt, mask, destination, threshold, cam_order):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Patch
    folder = destination / f'{index:05d}_{dataset.data_infos[index]["token"]}'
    folder.mkdir(parents=True,exist_ok=True)
    info = dataset.data_infos[index]
    det = result['pts_bbox']
    keep = det['scores_3d'].numpy() >= threshold
    corners = det['boxes_3d'].corners.numpy()[keep]
    det_labels = det['labels_3d'].numpy()[keep]
    keycam = info['cams'][dataset.ego_cam]
    key_to_global = transform(keycam['ego2global_rotation'],keycam['ego2global_translation'])
    panels = []
    for name in cam_order:
        cam = info['cams'][name]
        filename = Path(cam['data_path'])
        if not filename.is_file():
            filename = Path(dataset.data_root)/str(filename).split('nuscenes/')[-1]
        image = cv2.imread(str(filename))
        if image is None:
            raise FileNotFoundError(filename)
        cam_to_global = transform(cam['ego2global_rotation'],cam['ego2global_translation']) @ transform(cam['sensor2ego_rotation'],cam['sensor2ego_translation'])
        key_to_cam = np.linalg.inv(cam_to_global) @ key_to_global
        for box, label in zip(corners,det_labels):
            xyz = np.c_[box,np.ones(8)] @ key_to_cam.T
            xyh = xyz[:,:3] @ np.asarray(cam['cam_intrinsic']).T
            xy = xyh[:,:2]/np.maximum(xyh[:,2:3],1e-6)
            color = tuple(int(v) for v in PALETTE[(int(label)+1)%17][::-1])
            for a,b in EDGES:
                if xyz[a,2] > .5 and xyz[b,2] > .5:
                    # Clip before converting coordinates to OpenCV integers.
                    pa,pb = np.clip(xy[[a,b]],-100000,100000).astype(int)
                    visible,p1,p2 = cv2.clipLine((0,0,image.shape[1],image.shape[0]),tuple(pa),tuple(pb))
                    if visible: cv2.line(image,p1,p2,color,3)
        cv2.putText(image,name,(20,45),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        panels.append(cv2.resize(image,(640,360)))
    cv2.imwrite(str(folder/'det_cameras.jpg'),np.vstack([np.hstack(panels[:3]),np.hstack(panels[3:])]))
    fig,axes = plt.subplots(1,3,figsize=(18,6))
    for box,label in zip(corners,det_labels):
        axes[0].add_patch(Polygon(box[[0,3,7,4],:2],fill=False,color=PALETTE[(int(label)+1)%17]/255.,linewidth=1))
    # ann_infos used by BEVDet's LoadAnnotations are in key-ego coordinates.
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    gt_boxes = np.asarray(info['ann_infos'][0],dtype=np.float32)
    if len(gt_boxes):
        gt_corners = LiDARInstance3DBoxes(gt_boxes,box_dim=gt_boxes.shape[1],origin=(.5,.5,.5)).corners.numpy()
        for box in gt_corners:
            axes[0].add_patch(Polygon(box[[0,3,7,4],:2],fill=False,edgecolor='gray',linestyle='--',linewidth=.7))
    axes[0].set(xlim=(-51.2,51.2),ylim=(-51.2,51.2),xlabel='ego x (m)',ylabel='ego y (m)',title=f'Detection: color=pred >= {threshold}; dashed=GT')
    axes[0].set_aspect('equal');axes[0].grid(alpha=.2)
    for ax,labels,title in [(axes[1],result['occ'],'OCC prediction'),(axes[2],gt,'OCC ground truth')]:
        ax.imshow(occ_top(labels,mask),extent=(40,-40,-40,40),interpolation='nearest')
        ax.set(title=title+' (camera-visible, highest occupied voxel)',xlabel='ego y (m)',ylabel='ego x (m)')
    fig.tight_layout();fig.savefig(folder/'bev_comparison.png',dpi=130);plt.close(fig)
    fig = plt.figure(figsize=(14,6))
    for position,labels,title in [(1,result['occ'],'Prediction'),(2,gt,'Ground truth')]:
        xyz = np.argwhere(mask & (labels < 17))
        # Render at most 50000 centers for readability; NPZ retains all voxels.
        if len(xyz)>50000: xyz=xyz[::int(np.ceil(len(xyz)/50000))]
        ax = fig.add_subplot(1,2,position,projection='3d')
        if len(xyz):
            points=xyz*.4+np.array([-39.8,-39.8,-.8])
            ax.scatter(points[:,0],points[:,1],points[:,2],c=PALETTE[labels[tuple(xyz.T)]]/255.,s=1,depthshade=False)
        ax.set(xlim=(-40,40),ylim=(-40,40),zlim=(-1,5.4),xlabel='ego x (m)',ylabel='ego y (m)',zlabel='z (m)',title=title+' occupied voxel centers')
        ax.set_box_aspect((80,80,15));ax.view_init(elev=35,azim=-60)
    fig.tight_layout();fig.savefig(folder/'occ_3d.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,2));ax.axis('off')
    ax.legend(handles=[Patch(color=PALETTE[i]/255.,label=name) for i,name in enumerate(OCC_CLASSES[:17])],loc='center',ncol=6)
    fig.savefig(folder/'occ_legend.png',bbox_inches='tight',dpi=120);plt.close(fig)
    np.savez_compressed(folder/'occupancy.npz',prediction=result['occ'],ground_truth=gt,mask_camera=mask)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config');p.add_argument('checkpoint')
    p.add_argument('--out-dir',default='work_dirs/eval_epoch20')
    p.add_argument('--vis-count',type=int,default=10)
    p.add_argument('--vis-every',type=int,default=1)
    p.add_argument('--score-thr',type=float,default=.3)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--gpu-id',type=int,default=0)
    p.add_argument('--smoke-test',type=int,default=0,help='Only run N samples; no official evaluation')
    args=p.parse_args()
    if args.vis_every<1 or args.vis_count<0 or args.smoke_test<0: p.error('Invalid sample counts')
    cfg=Config.fromfile(args.config)
    cfg.data.test.test_mode=True
    cfg.model.train_cfg=None;cfg.model.img_backbone.pretrained=None
    dataset=build_dataset(cfg.data.test)
    # Fail early on missing annotations, before loading weights or running GPU.
    for i in range(len(dataset)):
        if not Path(dataset._occ_path(i)).is_file(): raise FileNotFoundError(dataset._occ_path(i))
    output=Path(args.out_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    model=build_model(cfg.model,test_cfg=cfg.get('test_cfg'))
    checkpoint=torch.load(args.checkpoint,map_location='cpu')
    state=checkpoint.get('state_dict',checkpoint)
    model.load_state_dict({k[7:] if k.startswith('module.') else k:v for k,v in state.items()},strict=True)
    model.CLASSES=dataset.CLASSES
    model=MMDataParallel(model.cuda(args.gpu_id),device_ids=[args.gpu_id]);model.eval()
    loader=build_dataloader(dataset,samples_per_gpu=1,workers_per_gpu=args.workers,dist=False,shuffle=False)
    hist=np.zeros((18,18),dtype=np.int64);detections=[];visualized=0
    total=min(args.smoke_test,len(dataset)) if args.smoke_test else len(dataset)
    progress=mmcv.ProgressBar(total)
    for index,data in enumerate(loader):
        with torch.no_grad(): result=model(return_loss=False,rescale=True,**data)[0]
        pred=np.asarray(result['occ'])
        with np.load(dataset._occ_path(index)) as label:
            gt=label['semantics'];mask=label['mask_camera'].astype(bool) & (gt<18)
        if pred.shape!=gt.shape or np.any((pred<0)|(pred>=18)):raise ValueError('Invalid occupancy predictions')
        hist+=np.bincount(18*gt[mask].astype(np.int64)+pred[mask].astype(np.int64),minlength=324).reshape(18,18)
        detections.append({'pts_bbox':result['pts_bbox']})
        if visualized<args.vis_count and index%args.vis_every==0:
            save_vis(dataset,index,result,gt,mask,output/'visualizations',args.score_thr,cfg.data_config['cams'])
            visualized+=1
        progress.update()
        if index+1>=total:break
    if args.smoke_test:
        print('\nSmoke test passed; official metrics skipped. Visualizations:',output/'visualizations')
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
    det_metrics=dataset.evaluate(detections,metric='bbox',jsonfile_prefix=str(output/'detection'))
    metrics.update({k:float(v) for k,v in det_metrics.items()})
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print('All metrics:',json.dumps(metrics,indent=2));print('Results:',output)

if __name__=='__main__':main()
