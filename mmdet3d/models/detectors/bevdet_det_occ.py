# Copyright (c) OpenMMLab. All rights reserved.
from .bevdet import BEVDet
from ..builder import DETECTORS, build_head, build_neck


@DETECTORS.register_module()
class BEVDetDetOcc(BEVDet):
    """Single-frame detection + FlashOCC sharing the complete BEV encoder."""
    def __init__(self, occ_adapter, occ_head, occ_loss_weight=1., **kwargs):
        super().__init__(**kwargs)
        if occ_loss_weight < 0:
            raise ValueError('occ_loss_weight must be nonnegative')
        self.occ_adapter = build_neck(occ_adapter)
        self.occ_head = build_head(occ_head)
        self.occ_loss_weight = occ_loss_weight

    def forward_train(self, points=None, img_metas=None, gt_bboxes_3d=None,
                      gt_labels_3d=None, img_inputs=None, voxel_semantics=None,
                      mask_camera=None, gt_bboxes_ignore=None, **kwargs):
        if voxel_semantics is None or mask_camera is None:
            raise ValueError('Joint training requires voxel_semantics and mask_camera')
        # Exactly one image/LSS/BEV-encoder pass for both tasks.
        feats, _, _ = self.extract_feat(points, img_inputs, img_metas, **kwargs)
        losses = self.forward_pts_train(feats, gt_bboxes_3d, gt_labels_3d,
                                        img_metas, gt_bboxes_ignore)
        occ_pred = self.occ_head(self.occ_adapter(feats[0]))
        occ_losses = self.occ_head.loss(occ_pred, voxel_semantics, mask_camera)
        losses.update({key: value * self.occ_loss_weight
                       for key, value in occ_losses.items()})
        return losses

    def simple_test(self, points, img_metas, img=None, rescale=False, **kwargs):
        feats, _, _ = self.extract_feat(points, img, img_metas, **kwargs)
        boxes = self.simple_test_pts(feats, img_metas, rescale=rescale)
        occ = self.occ_head.get_occ(self.occ_head(self.occ_adapter(feats[0])))
        return [dict(pts_bbox=box, occ=labels) for box, labels in zip(boxes, occ)]
