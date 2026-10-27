"""Opt-in identical CenterHead target algorithm on CPU, avoiding per-box CUDA sync.

Inference modules and checkpoint keys are unchanged. Only non-differentiable
training target generation moves to CPU, followed by bulk tensor transfers.
"""
import torch
from mmdet.models import HEADS
from .centerpoint_head import CenterHead


@HEADS.register_module()
class CenterHeadCPUTargets(CenterHead):
    @torch.no_grad()
    def get_targets(self, gt_bboxes_3d, gt_labels_3d):
        cached = getattr(self, '_precomputed_targets', None)
        if cached is not None:
            if len(cached) != 4 or any(len(g) != len(self.task_heads) for g in cached):
                raise ValueError('Malformed precomputed target groups')
            if any(t.device != gt_labels_3d[0].device for g in cached for t in g):
                raise ValueError('Precomputed targets are on the wrong device')
            return tuple(list(g) for g in cached)
        if not gt_labels_3d:
            raise ValueError('Cannot construct targets for an empty batch')
        device = gt_labels_3d[0].device
        if any(t.device != device for t in gt_labels_3d):
            raise ValueError('All labels must be on the same device')
        boxes = [b.to('cpu') for b in gt_bboxes_3d]
        labels = [t.detach().cpu() for t in gt_labels_3d]
        targets = super().get_targets(boxes, labels)
        return tuple([t.to(device=device) for t in task] for task in targets)
