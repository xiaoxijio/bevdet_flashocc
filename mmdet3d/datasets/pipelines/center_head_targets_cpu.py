"""Opt-in worker-side CenterHead targets using the unmodified reference algorithm."""
from types import SimpleNamespace
import torch
from mmcv.parallel import DataContainer
from ..builder import PIPELINES
from mmdet3d.models.dense_heads.centerpoint_head import CenterHead


@PIPELINES.register_module()
class CenterHeadTargetsCPU:
    def __init__(self, train_cfg, tasks, norm_bbox=True, with_velocity=True):
        self.spec = SimpleNamespace(train_cfg=train_cfg,
            class_names=[t['class_names'] for t in tasks],
            task_heads=[None] * len(tasks), norm_bbox=norm_bbox,
            with_velocity=with_velocity)

    @torch.no_grad()
    def __call__(self, results):
        boxes = results['gt_bboxes_3d']
        labels = results['gt_labels_3d']
        if isinstance(boxes, DataContainer): boxes = boxes.data
        if isinstance(labels, DataContainer): labels = labels.data
        labels = torch.as_tensor(labels, dtype=torch.long, device='cpu')
        values = CenterHead.get_targets_single(self.spec, boxes.to('cpu'), labels)
        results['det_targets'] = tuple(
            [DataContainer(t, stack=True, pad_dims=None) for t in task] for task in values)
        return results
