"""Headless joint visualizations: BEVDet camera/BEV context + FlashOCC voxels.

References (rendering implemented locally, no OpenGL runtime required):
https://github.com/HuangJunJie2017/BEVDet/blob/dev3.0/tools/analysis_tools/vis.py
https://github.com/Yzichen/FlashOCC/blob/master/tools/analysis_tools/vis_occ.py
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
# FlashOCC/Occ3D RGB semantic palette; class 17 is free space.
PALETTE = np.array([[0,0,0],[112,128,144],[220,20,60],[255,127,80],
 [255,158,0],[233,150,70],[255,61,99],[0,0,230],[47,79,79],
 [255,140,0],[255,99,71],[0,207,191],[175,0,75],[75,0,75],
 [112,180,60],[222,184,135],[0,175,0],[248,249,251]], dtype=np.uint8)
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
                   labels=None, classes=None, width=640, points=None):
    panel = np.full((640, width, 3), 250, dtype=np.uint8)
    panel[:42] = (40,31,24)
    text(panel, title, (16,28), .62)
    left, top, side = 48, 64, width - 96
    if occupancy is not None:
        rgb = occupancy
        panel[top:top+side, left:left+side] = cv2.resize(
            rgb[..., ::-1], (side, side), interpolation=cv2.INTER_NEAREST)
    if points is not None:
        points = np.asarray(points)
        valid = np.isfinite(points).all(1) & (np.abs(points[:,:2]) < bound).all(1) & (points[:,2] > -3) & (points[:,2] < 4)
        points = points[valid]
        if len(points):
            uv = ego_pixels(points[:,:2], bound, side)
            intensity = np.clip((points[:,2]+2.5)/5., 0, 1)
            colors = np.stack([210-50*intensity, 205-65*intensity, 190-85*intensity], axis=1).astype(np.uint8)
            panel[top+uv[:,1], left+uv[:,0]] = colors
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
        image=cv2.resize(image,(628,348), interpolation=cv2.INTER_AREA)
        overlay = image.copy()
        cv2.rectangle(overlay,(0,0),(628,32),(40,31,24),-1)
        image = cv2.addWeighted(overlay,.85,image,.15,0)
        text(image,name.replace('CAM_','').replace('_',' '),(14,23),.55)
        card = np.full((360,640,3), (248,249,251), dtype=np.uint8)
        card[6:354,6:634] = image
        panels.append(card)
    return np.vstack([np.hstack(panels[:3]),np.hstack(panels[3:])])


def occupancy_3d_panel(title, labels, mask, bound=40., width=640):
    """Headless orthographic surface-voxel renderer; FlashOCC semantic palette.

    Draw exposed -X/-Y/+Z voxel faces with a painter's depth order, rather than
    flattening each column to one label. No OpenGL/display/Mayavi dependency.
    Physical axes match Occ3D: 0.4 m voxels, z starts at -1 m.
    """
    panel = np.full((640, width, 3), (248, 249, 251), dtype=np.uint8)
    panel[:42] = (40, 31, 24)
    text(panel, title, (18, 28), .62)
    shape = np.asarray(labels.shape)
    if labels.ndim != 3 or mask.shape != labels.shape:
        raise ValueError('Expected matching occupancy and mask [X,Y,Z]')
    sx, sy = 2 * bound / shape[0], 2 * bound / shape[1]
    voxel = np.array([sx, sy, .4])
    origin = np.array([-bound, -bound, -1.])
    azimuth, elevation = np.deg2rad(225.), np.deg2rad(27.)
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.])
    up = np.array([-np.cos(azimuth)*np.sin(elevation), -np.sin(azimuth)*np.sin(elevation), np.cos(elevation)])
    camera = np.array([np.cos(azimuth)*np.cos(elevation), np.sin(azimuth)*np.cos(elevation), np.sin(elevation)])
    basis = np.stack([right, -up], axis=1)
    extent = np.array([[x, y, z] for x in (-bound, bound) for y in (-bound, bound)
                       for z in (-1., -1. + shape[2]*.4)]) @ basis
    lower, upper = extent.min(0), extent.max(0)
    scale = min((width-48)/(upper[0]-lower[0]), 514/(upper[1]-lower[1]))
    offset = np.array([(width-scale*(upper[0]-lower[0]))/2, 66 + (514-scale*(upper[1]-lower[1]))/2])
    def project(points):
        return np.rint((np.asarray(points) @ basis-lower)*scale+offset).astype(np.int32)
    floor = project([[-bound,-bound,-1], [bound,-bound,-1], [bound,bound,-1], [-bound,bound,-1]])
    cv2.fillConvexPoly(panel, floor, (232, 235, 239), cv2.LINE_AA)
    for distance in np.arange(np.ceil(-bound/10)*10, bound+.01, 10):
        for points in ([[-bound,distance,-1],[bound,distance,-1]],
                       [[distance,-bound,-1],[distance,bound,-1]]):
            a, b = project(points)
            cv2.line(panel, tuple(a), tuple(b), (213, 218, 225), 1, cv2.LINE_AA)
    solid = mask.astype(bool) & (labels > 0) & (labels < 17)
    face_specs = [
        (0, -1, [[0,0,0],[0,1,0],[0,1,1],[0,0,1]], .76),
        (1, -1, [[0,0,0],[1,0,0],[1,0,1],[0,0,1]], .88),
        (2, 1, [[0,0,1],[1,0,1],[1,1,1],[0,1,1]], 1.),
    ]
    polygons, colors, depths = [], [], []
    for axis, direction, corners, shade in face_specs:
        neighbor = np.zeros_like(solid)
        src, dst = [slice(None)]*3, [slice(None)]*3
        if direction < 0:
            dst[axis], src[axis] = slice(1,None), slice(None,-1)
        else:
            dst[axis], src[axis] = slice(None,-1), slice(1,None)
        neighbor[tuple(dst)] = solid[tuple(src)]
        indices = np.argwhere(solid & ~neighbor)
        if not len(indices):
            continue
        vertices = (indices[:,None,:]+np.asarray(corners)[None,:,:])*voxel+origin
        polygons.append(project(vertices))
        rgb = PALETTE[labels[tuple(indices.T)]]
        colors.append(np.rint(rgb[:,::-1]*shade).astype(np.uint8))
        depths.append(vertices.mean(1) @ camera)
    if polygons:
        polygons, colors, depths = np.concatenate(polygons), np.concatenate(colors), np.concatenate(depths)
        for i in np.argsort(depths, kind='stable'):
            cv2.fillConvexPoly(panel, polygons[i], tuple(int(c) for c in colors[i]), cv2.LINE_AA)
    # A small physical compass, independent of semantic content.
    anchor = np.array([width-80., 579.])
    for vector, label, color in (([1,0,0], '+X', (73,94,200)),
                                  ([0,1,0], '+Y', (97,153,61)),
                                  ([0,0,1], '+Z', (200,132,65))):
        endpoint = np.rint(anchor + np.asarray(vector) @ basis*33).astype(int)
        cv2.arrowedLine(panel, tuple(anchor.astype(int)), tuple(endpoint), color, 2, cv2.LINE_AA, tipLength=.2)
        text(panel, label, tuple(endpoint+[-8,-5]), .36, color)
    text(panel, f'3D semantic voxels  |  +/-{bound:g} m  |  0.4 m height steps',
         (22, 626), .43, (95, 90, 84))
    return panel


def lidar_context(dataset, info):
    """LiDAR -> the same reference ego frame used by the joint detector."""
    filename = Path(info.get('lidar_path', ''))
    if not filename.is_file() and info.get('lidar_path'):
        filename = Path(dataset.data_root)/str(filename).split('nuscenes/')[-1]
    if not filename.is_file():
        return None
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 5)[:, :3]
    lidar_global = transform(info['ego2global_rotation'], info['ego2global_translation']) @ transform(info['lidar2ego_rotation'], info['lidar2ego_translation'])
    key = info['cams'][dataset.ego_cam]
    key_global = transform(key['ego2global_rotation'], key['ego2global_translation'])
    matrix = np.linalg.inv(key_global) @ lidar_global
    return (np.c_[points, np.ones(len(points))] @ matrix.T)[:, :3]


def compose_frame(dataset,index,result,gt,mask,frame_number,threshold=.3,occ_view='3d',backend='pth'):
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    info=dataset.data_infos[index]
    det=result['pts_bbox']
    scores=as_numpy(det['scores_3d'])
    keep=scores >= threshold
    corners=as_numpy(det['boxes_3d'].corners)[keep]
    labels=as_numpy(det['labels_3d'])[keep]
    ann=info.get('ann_infos', ([],[]))
    gt_boxes=np.asarray(ann[0],dtype=np.float32)
    gt_corners=None
    if len(gt_boxes):
        gt_corners=LiDARInstance3DBoxes(gt_boxes,box_dim=gt_boxes.shape[1], origin=(.5,.5,.5)).corners.numpy()
    crop=getattr(dataset,'occ_crop',None)
    occ_bound=40. if crop is None else (crop[1]-crop[0])*.4/2
    cameras=camera_mosaic(dataset,info,corners,labels)
    detection=draw_bev_panel('DETECTION / BEV',boxes=corners,gt_boxes=gt_corners,
                            labels=labels,classes=dataset.CLASSES,points=lidar_context(dataset,info))
    if occ_view == '3d':
        occ_panels=[occupancy_3d_panel(title, labels_, mask, occ_bound)
                    for title,labels_ in [('OCC / PREDICTION',result['occ']),('OCC / GROUND TRUTH',gt)]]
    else:
        occ_panels=[draw_bev_panel(title,bound=occ_bound,occupancy=occupancy_rgb(labels_,mask))
                    for title,labels_ in [('OCC / PREDICTION',result['occ']),('OCC / GROUND TRUTH',gt)]]
    header=np.full((58,1920,3),(40,31,24),dtype=np.uint8)
    text(header,'BEVDet + FlashOCC',(20,25),.72,(250,250,250),2)
    text(header,f'{backend.upper()}  /  {scene_name(info)}  /  FRAME {frame_number:04d}',(20,47),.43,(198,192,182))
    text(header,f'{len(corners)} detections  |  score >= {threshold:.2f}',(1350,28),.52,(220,217,210))
    text(header,str(info['token']), (1350,47), .4,(175,173,167))
    legend=np.full((108,1920,3),(248,249,251),dtype=np.uint8)
    for i,name in enumerate(OCC_NAMES[:17]):
        row,col=divmod(i,6);u,v=col*320+20,row*30+9
        cv2.rectangle(legend,(u,v),(u+13,v+13),tuple(int(x) for x in PALETTE[i][::-1]),-1)
        text(legend,name.replace('_',' '),(u+22,v+12),.4,(85,80,72))
    text(legend,'Detection: colored prediction / gray GT    |    OCC: visibility mask shared by prediction and GT',
         (20,103),.35,(125,119,109))
    return np.ascontiguousarray(np.vstack([header,cameras,np.hstack([detection]+occ_panels),legend]))


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
    def __init__(self,destination,threshold=.3,fps=2.,video=True,save_occupancy=False,occ_view='3d',backend='pth',mask_mode='camera'):
        self.destination=Path(destination)
        self.destination.mkdir(parents=True,exist_ok=True)
        self.mask_mode=mask_mode
        self.occ_view=occ_view
        self.backend=backend
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
        image=compose_frame(dataset,index,result,gt,mask,number,self.threshold,self.occ_view,self.backend)
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
                                prediction=result['occ'],ground_truth=gt,mask_render=mask)
        self.records.append(dict(scene=scene,frame=number,dataset_index=index,
                                 timestamp=int(timestamp),token=info['token'],image=path.name))
        return path

    def close(self):
        try:
            for writer in self.writers.values():
                writer.close()
        finally:
            manifest=dict(backend=self.backend,occ_view=self.occ_view,mask_mode=self.mask_mode,fps=self.fps,frames=self.records,scene_frames=dict(self.counts),
                          camera_order=CAMERAS,orientation='BEV: ego +x up, +y left; 3D: physical axes shown',
                          distance_rings_m=[10,20,30,40],origin_marker='small cross; no ego vehicle drawn')
            (self.destination/'index.json').write_text(json.dumps(manifest,indent=2))
