from pathlib import Path
import torch
from mmcv import Config
from mmcv.parallel import DataContainer, collate
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.models import build_model
from mmdet3d.models.dense_heads.centerpoint_head import CenterHead
from mmdet3d.datasets.pipelines.center_head_targets_cpu import CenterHeadTargetsCPU


def test_worker_targets_match_reference_and_collate():
    c=Config.fromfile('configs/bevdet/ep1_bevdet+flashocc/bevdet-b0-det-occ-g5-worker-targets-base.py')
    model=build_model(c.model)
    original=Config.fromfile('configs/bevdet/ep1_bevdet+flashocc/bevdet-b0-det-occ-g5-official.py')
    assert model.state_dict().keys()==build_model(original.model).state_dict().keys()
    generator=CenterHeadTargetsCPU(c.model.train_cfg.pts,c.model.pts_bbox_head.tasks)
    for boxes,labels in [
        (torch.tensor([[1.,2.,0.,4.,2.,1.5,.4,.1,.2],[-51.,-50.,-1.,.4,.5,1.8,-1.,0.,0.]]),torch.tensor([0,8])),
        (torch.empty(0,9),torch.empty(0,dtype=torch.long))]:
        boxes=LiDARInstance3DBoxes(boxes,box_dim=9)
        sample=generator(dict(gt_bboxes_3d=boxes,gt_labels_3d=labels))
        batched=collate([dict(det_targets=sample['det_targets'])]*2,samples_per_gpu=2)['det_targets']
        cached=tuple([dc.data[0] for dc in group] for group in batched)
        expected=CenterHead.get_targets(model.pts_bbox_head,[boxes,boxes],[labels,labels])
        for a,b in zip(expected,cached):
            for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0,equal_nan=True)
        model.pts_bbox_head._precomputed_targets=cached
        actual=model.pts_bbox_head.get_targets([boxes,boxes],[labels,labels])
        for a,b in zip(expected,actual):
            for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0,equal_nan=True)
        model.pts_bbox_head._precomputed_targets=None
