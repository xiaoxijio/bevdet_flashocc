"""Geometry and scene-order regressions for continuous joint visualizations."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import bev_occ_visualization as vis


def test_scene_selection_does_not_mix_time_or_scenes():
    def info(scene, timestamp):
        return dict(occ_path=f'gts/{scene}/token', scene_token=scene+'token',
                    timestamp=timestamp)
    infos=[info('scene-z',20),info('scene-a',1),info('scene-z',10),info('scene-a',2)]
    assert vis.select_frames(infos,scene_count=0) == [1,3,2,0]
    assert vis.select_frames(infos,scenes=['scene-z']) == [2,0]
    assert vis.select_frames(infos,scene_count=0,every=2) == [1,2]
    assert vis.select_frames(infos,scene_count=0,limit=3) == [1,3,2]
    with pytest.raises(ValueError):vis.select_frames(infos,scenes=['missing'])


def test_front_and_left_match_between_occupancy_and_detection():
    center,front,left=vis.ego_pixels([[0,0],[10,0],[0,10]],40,201)
    assert tuple(center)==(100,100)
    assert front[1]<center[1] and front[0]==center[0]
    assert left[0]<center[0] and left[1]==center[1]
    labels=np.full((4,4,2),17,dtype=np.uint8)
    labels[3,3,0]=4
    labels[3,3,1]=10  # highest occupied voxel wins
    mask=np.ones_like(labels,dtype=bool)
    rgb=vis.occupancy_rgb(labels,mask)
    np.testing.assert_array_equal(rgb[0,0],vis.PALETTE[10])
    mask[3,3,1]=False
    np.testing.assert_array_equal(vis.occupancy_rgb(labels,mask)[0,0],vis.PALETTE[4])


def test_composite_and_flat_outputs_contain_distance_rings_and_manifest(tmp_path,monkeypatch):
    monkeypatch.setattr(vis,'camera_mosaic',lambda *a:np.zeros((720,1920,3),dtype=np.uint8))
    dataset=SimpleNamespace(CLASSES=['car'],data_infos=[dict(
        occ_path='gts/scene-0001/token',scene_token='scene-token',timestamp=1000000,
        token='token',ann_infos=([],[]))])
    result=dict(pts_bbox=dict(scores_3d=np.array([]),labels_3d=np.array([],dtype=int),
        boxes_3d=SimpleNamespace(corners=np.empty((0,8,3)))),
        occ=np.full((200,200,16),17,dtype=np.uint8))
    mask=np.ones_like(result['occ'],dtype=bool)
    renderer=vis.SceneRenderer(tmp_path,video=False)
    path=renderer.save(dataset,0,result,result['occ'],mask)
    renderer.close()
    assert path.parent==tmp_path and path.name=='scene-0001_00000.jpg'
    assert not any(p.is_dir() for p in tmp_path.iterdir())
    image=vis.compose_frame(dataset,0,result,result['occ'],mask,0)
    assert image.shape==(1526,1920,3)
    # Only the detection panel has rings now; the two OCC panels are 3D.
    # Accept antialiased strokes instead of requiring an exact pixel color.
    plot = image[842:1386, 48:592].astype(np.int16)
    assert np.count_nonzero(np.max(np.abs(plot-np.asarray(vis.RING_COLOR)),axis=-1) < 30) > 100
    import json
    assert json.loads((tmp_path/'index.json').read_text())['scene_frames']=={'scene-0001':1}


def test_distance_rings_use_metric_radii(monkeypatch):
    calls=[]
    original=vis.cv2.circle
    def circle(image,center,radius,*args,**kwargs):
        calls.append((center,radius))
        return original(image,center,radius,*args,**kwargs)
    monkeypatch.setattr(vis.cv2,'circle',circle)
    image=np.full((640,640,3),250,dtype=np.uint8)
    vis.draw_distance_rings(image,40.,544,(48,64))
    expected_center=tuple(vis.ego_pixels([[0,0]],40.,544)[0])
    assert calls == [(expected_center,round(r*543/80)) for r in (10,20,30,40)]
