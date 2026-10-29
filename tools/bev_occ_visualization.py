"""Headless, scene-continuous joint Det/OCC visualization.

Layout inspired by Sparse4D's NuScenes3DDetTrackDataset.show: labelled six-view
mosaic, BEV context, ordered images and video. Geometry/rendering here is local
implementation; no Sparse4D runtime dependency. RGB palettes, BGR OpenCV frames.
"""
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pyquaternion import Quaternion

OCC_NAMES = ('others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
             'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
             'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
             'manmade', 'vegetation', 'free')
PALETTE = np.array([[90,90,90],[255,120,50],[255,192,203],[255,255,0],
 [0,150,245],[0,255,255],[200,180,0],[255,0,0],[255,240,150],
 [135,60,0],[160,32,240],[255,0,255],[175,0,75],[75,0,75],
 [150,240,80],[230,230,250],[0,175,0],[245,245,245]], dtype=np.uint8)
CAMERAS = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
           'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT')
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
RING_COLOR = (125, 125, 125)  # BGR metric distance guides, like Sparse4D.


def as_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def transform(rotation, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(rotation).rotation_matrix
    matrix[:3, 3] = translation
    return matrix


def scene_name(info):
    if info.get('occ_path'):
        return Path(info['occ_path']).parent.name
    return str(info['scene_token'])


def select_frames(infos, scenes=None, scene_count=1, every=1, limit=0):
    """Select complete scenes, preserving chronological frame order within each."""
    groups = defaultdict(list)
    for index, info in enumerate(infos):
        groups[scene_name(info)].append(index)
    for indices in groups.values():
        indices.sort(key=lambda i: infos[i]['timestamp'])
    ordered = sorted(groups, key=lambda name: infos[groups[name][0]]['timestamp'])
    if scenes:
        selected = [name for name in ordered if name in scenes or
                    str(infos[groups[name][0]]['scene_token']) in scenes]
        matched = set(selected) | {str(infos[groups[n][0]]['scene_token']) for n in selected}
        missing = set(scenes) - matched
        if missing:
            raise ValueError('Unknown scene names/tokens: ' + ', '.join(sorted(missing)))
    else:
        selected = ordered[:scene_count] if scene_count else ordered
    indices = [i for name in selected for i in groups[name][::every]]
    return indices[:limit] if limit else indices


def ego_pixels(xy, bound, size, offset=(0, 0)):
    """Metric ego x forward/up, y left/left -> image column,row."""
    xy = np.asarray(xy)
    uv = np.stack(((bound - xy[..., 1]), (bound - xy[..., 0])), axis=-1)
    return np.rint(uv * ((size - 1) / (2 * bound)) + offset).astype(np.int32)


def occupancy_rgb(labels, mask):
    valid = mask.astype(bool) & (labels >= 0) & (labels < 17)
    heights = np.where(valid, np.arange(labels.shape[2])[None, None], -1).max(2)
    top = np.take_along_axis(labels, np.maximum(heights, 0)[..., None], 2)[..., 0]
    top = np.where(heights >= 0, top, 17)
    rgb = PALETTE[top]
    rgb[heights < 0] = (245, 245, 245)
    # labels[X,Y,Z] -> screen row=-X, column=-Y (do not transpose XY).
    return np.ascontiguousarray(rgb[::-1, ::-1])


def color_for_detection(label, classes):
    name = classes[int(label)]
    return tuple(int(v) for v in PALETTE[OCC_NAMES.index(name)][::-1])


def text(image, label, position, scale=.6, color=(235,235,235), thickness=1):
    cv2.putText(image, str(label), position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_distance_rings(image, bound, size, offset):
    """Concentric 10m distance rings and origin cross, without an ego car."""
    left, top = offset
    roi = image[top:top+size, left:left+size]
    center = tuple(ego_pixels([[0, 0]], bound, size)[0])
    cv2.line(roi, (0, center[1]), (size-1, center[1]), RING_COLOR, 1, cv2.LINE_AA)
    cv2.line(roi, (center[0], 0), (center[0], size-1), RING_COLOR, 1, cv2.LINE_AA)
    for radius_m in range(10, int(bound)+1, 10):
        radius_px = round(radius_m * (size-1) / (2 * bound))
        cv2.circle(roi, center, radius_px, RING_COLOR, 1, cv2.LINE_AA)
        # Label each ring in the upper-right quadrant; identical metric scale in all panels.
        uv = ego_pixels([[radius_m / np.sqrt(2), -radius_m / np.sqrt(2)]],
                        bound, size)[0]
        label = f'{radius_m}m'
        width, height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .43, 1)[0]
        x, y = int(uv[0]-width/2), int(uv[1])
        cv2.rectangle(roi, (x-2, y-height-2), (x+width+2, y+3), (250,250,250), -1)
        text(roi, label, (x,y), .43, (65,65,65))
    cv2.drawMarker(roi, center, (40,40,40), cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)


def draw_bev_panel(title, bound=40., occupancy=None, boxes=None, gt_boxes=None,
                   labels=None, classes=None, width=640):
    panel = np.full((640, width, 3), 250, dtype=np.uint8)
    panel[:42] = (35,35,35)
    text(panel, title, (16,28), .62)
    left, top, side = 48, 64, width - 96
    if occupancy is not None:
        rgb = occupancy
        panel[top:top+side, left:left+side] = cv2.resize(
            rgb[..., ::-1], (side, side), interpolation=cv2.INTER_NEAREST)
    draw_distance_rings(panel,bound,side,(left,top))
    cv2.rectangle(panel,(left,top),(left+side-1,top+side-1),(90,90,90),1)
    if gt_boxes is not None:
        for box in gt_boxes:
            poly=ego_pixels(box[[0,3,7,4],:2],bound,side,(left,top))
            # Clip drawing to the plot, even when targets lie outside its range.
            roi=panel[top:top+side,left:left+side]
            cv2.polylines(roi,[poly-[left,top]],True,(135,135,135),1,cv2.LINE_AA)
    if boxes is not None:
        roi=panel[top:top+side,left:left+side]
        for box,label in zip(boxes,labels):
            poly=ego_pixels(box[[0,3,7,4],:2],bound,side)
            color=color_for_detection(label,classes)
            cv2.polylines(roi,[poly],True,color,2,cv2.LINE_AA)
            # LiDARInstance3DBoxes local +x points along the box heading.
            start=box.mean(0)[:2]
            front=box[[4,5,6,7]].mean(0)[:2]
            a,b=ego_pixels([start,front],bound,side)
            cv2.line(roi,tuple(a),tuple(b),color,2,cv2.LINE_AA)
    text(panel,f'FRONT +x up  |  LEFT +y left  |  range +/-{bound:g} m',
         (28,635),.43,(65,65,65))
    return panel


def camera_mosaic(dataset, info, corners, labels):
    key = info['cams'][dataset.ego_cam]
    key_to_global = transform(key['ego2global_rotation'], key['ego2global_translation'])
    panels=[]
    for name in CAMERAS:
        cam=info['cams'][name]
        filename=Path(cam['data_path'])
        if not filename.is_file():
            filename=Path(dataset.data_root)/str(filename).split('nuscenes/')[-1]
        image=cv2.imread(str(filename))
        if image is None:
            raise FileNotFoundError(filename)
        cam_to_global=(transform(cam['ego2global_rotation'],cam['ego2global_translation']) @
                       transform(cam['sensor2ego_rotation'],cam['sensor2ego_translation']))
        key_to_cam=np.linalg.inv(cam_to_global) @ key_to_global
        intrinsic=np.asarray(cam['cam_intrinsic'])
        for box,label in zip(corners,labels):
            xyz=np.c_[box,np.ones(8)] @ key_to_cam.T
            color=color_for_detection(label,dataset.CLASSES)
            for a,b in EDGES:
                pa,pb=xyz[a,:3].copy(),xyz[b,:3].copy()
                if pa[2]<.5 and pb[2]<.5:
                    continue
                if pa[2]<.5:
                    pa=pa+(pb-pa)*((.5-pa[2])/(pb[2]-pa[2]))
                if pb[2]<.5:
                    pb=pb+(pa-pb)*((.5-pb[2])/(pa[2]-pb[2]))
                uv=np.stack([pa,pb]) @ intrinsic.T
                uv=uv[:,:2]/uv[:,2:3]
                if not np.isfinite(uv).all():
                    continue
                uv=np.clip(uv,-100000,100000).astype(int)
                visible,p1,p2=cv2.clipLine((0,0,image.shape[1],image.shape[0]),
                                          tuple(uv[0]),tuple(uv[1]))
                if visible:
                    cv2.line(image,p1,p2,color,3,cv2.LINE_AA)
        image=cv2.resize(image,(640,360))
        cv2.rectangle(image,(0,0),(640,30),(30,30,30),-1)
        text(image,name.replace('CAM_',''),(15,22),.6)
        panels.append(image)
    return np.vstack([np.hstack(panels[:3]),np.hstack(panels[3:])])


def compose_frame(dataset,index,result,gt,mask,frame_number,threshold=.3):
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    info=dataset.data_infos[index]
    det=result['pts_bbox']
    keep=as_numpy(det['scores_3d']) >= threshold
    corners=as_numpy(det['boxes_3d'].corners)[keep]
    labels=as_numpy(det['labels_3d'])[keep]
    gt_boxes=np.asarray(info['ann_infos'][0],dtype=np.float32)
    gt_corners=None
    if len(gt_boxes):
        gt_corners=LiDARInstance3DBoxes(gt_boxes,box_dim=gt_boxes.shape[1],
                                      origin=(.5,.5,.5)).corners.numpy()
    crop=getattr(dataset,'occ_crop',None)
    occ_bound=40. if crop is None else (crop[1]-crop[0])*.4/2
    cameras=camera_mosaic(dataset,info,corners,labels)
    panels=[draw_bev_panel('DETECTION | color: prediction, gray: GT',boxes=corners,
                          gt_boxes=gt_corners,labels=labels,classes=dataset.CLASSES),
            draw_bev_panel('OCC PREDICTION | camera-visible',bound=occ_bound,occupancy=occupancy_rgb(result['occ'],mask)),
            draw_bev_panel('OCC GROUND TRUTH | camera-visible',bound=occ_bound,occupancy=occupancy_rgb(gt,mask))]
    header=np.full((58,1920,3),25,dtype=np.uint8)
    text(header,f'{scene_name(info)}  |  frame {frame_number:04d}  |  token {info["token"]}',(20,25),.65)
    text(header,'Single-frame Det + FlashOCC | distance rings: metric scale shown in each panel | occupied top voxel',(20,47),.46)
    legend=np.full((108,1920,3),245,dtype=np.uint8)
    for i,name in enumerate(OCC_NAMES[:17]):
        row,col=divmod(i,6);u,v=col*320+16,row*32+7
        cv2.rectangle(legend,(u,v),(u+18,v+18),tuple(int(x) for x in PALETTE[i][::-1]),-1)
        text(legend,name,(u+25,v+15),.43,(40,40,40))
    return np.ascontiguousarray(np.vstack([header,cameras,np.hstack(panels),legend]))


class SceneVideo:
    def __init__(self,path,shape,fps):
        self.path=path
        self.process=None
        self.writer=None
        height,width=shape[:2]
        if shutil.which('ffmpeg'):
            cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
                 '-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-an',
                 '-c:v','libx264','-threads','2','-preset','veryfast','-crf','20',
                 '-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
            self.process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        else:
            self.writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps,(width,height))
            if not self.writer.isOpened():
                raise RuntimeError('Cannot create video; install ffmpeg or use --no-video')
    def write(self,frame):
        if self.process:
            self.process.stdin.write(frame.tobytes())
        else:
            self.writer.write(frame)
    def close(self):
        if self.process:
            self.process.stdin.close()
            code=self.process.wait()
            if code:
                raise RuntimeError(f'ffmpeg failed for {self.path}: {code}')
        elif self.writer:
            self.writer.release()


class SceneRenderer:
    """Flat output directory, stable per-scene sequence numbers, streaming videos."""
    def __init__(self,destination,threshold=.3,fps=2.,video=True,save_occupancy=False):
        self.destination=Path(destination)
        self.destination.mkdir(parents=True,exist_ok=True)
        self.threshold=threshold
        self.fps=fps
        self.video=video
        self.save_occupancy=save_occupancy
        self.records=[]
        self.writers={}
        self.counts=defaultdict(int)
        self.last_timestamp={}
        self.expected_counts={}

    def save(self,dataset,index,result,gt,mask):
        info=dataset.data_infos[index]
        scene=scene_name(info)
        timestamp=info['timestamp']
        if scene in self.last_timestamp and timestamp<=self.last_timestamp[scene]:
            raise ValueError('Frames must be rendered chronologically within a scene')
        self.last_timestamp[scene]=timestamp
        number=self.counts[scene]
        self.counts[scene]+=1
        stem=f'{scene}_{number:05d}'
        image=compose_frame(dataset,index,result,gt,mask,number,self.threshold)
        path=self.destination/(stem+'.jpg')
        if not cv2.imwrite(str(path),image,[cv2.IMWRITE_JPEG_QUALITY,92]):
            raise IOError(path)
        if self.video:
            if scene not in self.writers:
                self.writers[scene]=SceneVideo(self.destination/(scene+'.mp4'),image.shape,self.fps)
            self.writers[scene].write(image)
            if self.counts[scene] == self.expected_counts.get(scene):
                self.writers.pop(scene).close()
        if self.save_occupancy:
            np.savez_compressed(self.destination/(stem+'_occupancy.npz'),
                                prediction=result['occ'],ground_truth=gt,mask_camera=mask)
        self.records.append(dict(scene=scene,frame=number,dataset_index=index,
                                 timestamp=int(timestamp),token=info['token'],image=path.name))
        return path

    def close(self):
        try:
            for writer in self.writers.values():
                writer.close()
        finally:
            manifest=dict(fps=self.fps,frames=self.records,scene_frames=dict(self.counts),
                          camera_order=CAMERAS,orientation='ego +x up, +y left',
                          distance_rings_m=[10,20,30,40],origin_marker='small cross; no ego vehicle drawn')
            (self.destination/'index.json').write_text(json.dumps(manifest,indent=2))
