# Copyright (c) OpenMMLab. All rights reserved.
import math
import logging
import torch
from mmcv.runner import _load_checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from .bevdet import BEVDet
from ..necks.view_transformer import LSSViewTransformerBEVDepth
from ..builder import DETECTORS, build_head, build_neck


@DETECTORS.register_module()
class BEVDetDetOcc(BEVDet):
    """Single-frame detection + FlashOCC sharing the complete BEV encoder."""
    def __init__(self, occ_adapter, occ_head, occ_loss_weight=1.,
                 depth_loss_weight=0., depth_loss_from_logits=False,
                 transfer_from=None, **kwargs):
        super().__init__(**kwargs)
        if occ_loss_weight < 0:
            raise ValueError('occ_loss_weight must be nonnegative')
        self.occ_adapter = build_neck(occ_adapter)
        self.occ_head = build_head(occ_head)
        self.occ_loss_weight = occ_loss_weight
        if not math.isfinite(depth_loss_weight) or depth_loss_weight < 0:
            raise ValueError('depth_loss_weight must be finite and nonnegative')
        self.depth_loss_weight = depth_loss_weight
        self.depth_loss_from_logits = depth_loss_from_logits
        self.transfer_from = transfer_from

    def init_weights(self):
        if self._is_init:
            return
        super().init_weights()
        if self.transfer_from is None:
            return
        # A new image backbone needs a fresh, small-scale FPN. Do not mix
        # new lateral kernels with the teacher's trained output convolution.
        for module in self.img_neck.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.reset_parameters()
        checkpoint = _load_checkpoint(self.transfer_from, map_location='cpu')
        source = checkpoint.get('state_dict', checkpoint)
        source = {k[7:] if k.startswith('module.') else k: v
                  for k, v in source.items()}
        target = self.state_dict()
        reused = {k: v for k, v in target.items()
                  if not k.startswith(('img_backbone.', 'img_neck.'))}
        incompatible = [k for k, v in reused.items()
                        if k not in source or source[k].shape != v.shape]
        if incompatible:
            raise ValueError('Incompatible transfer checkpoint: ' + ', '.join(incompatible[:10]))
        self.load_state_dict({k: source[k] for k in reused}, strict=False)
        logging.getLogger('mmdet').info(
            'Transferred %d downstream tensors from %s; kept ImageNet backbone '
            'and freshly initialized image FPN.', len(reused), self.transfer_from)

    @staticmethod
    def depth_bce_with_logits(logits, labels):
        """BCE(softmax(logits), labels), without saturated probabilities.

        Each class has log-odds z_i - logsumexp(z_j, j != i).
        Prefix/suffix logsumexp avoids subtracting nearly equal probabilities.
        This preserves the existing independent BCE objective, not CE.
        """
        if logits.shape[-1] < 2:
            raise ValueError('Depth BCE requires at least two bins')
        prefix = torch.logcumsumexp(logits, dim=-1)
        suffix = torch.logcumsumexp(logits.flip(-1), dim=-1).flip(-1)
        left = F.pad(prefix[..., :-1], (1, 0), value=-float('inf'))
        right = F.pad(suffix[..., 1:], (0, 1), value=-float('inf'))
        other = torch.logaddexp(left, right)
        return F.binary_cross_entropy_with_logits(
            logits - other, labels, reduction='sum')

    def get_depth_loss(self, gt_depth, depth):
        """Depth supervision using probabilities or opt-in stable logits BCE."""
        vt = self.img_view_transformer
        if gt_depth.ndim != 4 or depth.ndim != 4:
            raise ValueError('Expected gt_depth [B,N,H,W] and depth [B*N,D,h,w]')
        b, n, h, w = gt_depth.shape
        if h % vt.downsample or w % vt.downsample:
            raise ValueError('Depth label size must be divisible by LSS downsample')
        if depth.shape != (b*n, vt.D, h//vt.downsample, w//vt.downsample):
            raise ValueError('Depth label and prediction shapes do not match')
        with autocast(enabled=False):
            # Reuse the repository BEVDepth nearest-valid-point/binning rule.
            labels = LSSViewTransformerBEVDepth.get_downsampled_gt_depth(
                vt, gt_depth.float())
            preds = depth.float().permute(0, 2, 3, 1).reshape(-1, vt.D)
            valid = labels.max(dim=1).values > 0
            if not valid.any():
                return preds.sum() * 0.
            if self.depth_loss_from_logits:
                loss = self.depth_bce_with_logits(preds[valid], labels[valid])
            else:
                loss = F.binary_cross_entropy(
                    preds[valid], labels[valid], reduction='sum')
            return self.depth_loss_weight * loss / valid.sum().clamp_min(1)

    def forward_train(self, points=None, img_metas=None, gt_bboxes_3d=None,
                      gt_labels_3d=None, img_inputs=None, voxel_semantics=None,
                      mask_camera=None, gt_bboxes_ignore=None, gt_depth=None, **kwargs):
        if voxel_semantics is None or mask_camera is None:
            raise ValueError('Joint training requires voxel_semantics and mask_camera')
        # Exactly one image/LSS/BEV-encoder pass for both tasks.
        if self.depth_loss_weight > 0 and gt_depth is None:
            raise ValueError('Depth supervision requires gt_depth in train_pipeline')
        feats, _, depth = self.extract_feat(
            points, img_inputs, img_metas,
            return_depth_logits=self.depth_loss_from_logits, **kwargs)
        losses = self.forward_pts_train(feats, gt_bboxes_3d, gt_labels_3d,
                                        img_metas, gt_bboxes_ignore)
        occ_pred = self.occ_head(self.occ_adapter(feats[0]))
        occ_losses = self.occ_head.loss(occ_pred, voxel_semantics, mask_camera)
        losses.update({key: value * self.occ_loss_weight
                       for key, value in occ_losses.items()})
        if self.depth_loss_weight > 0:
            losses['loss_depth'] = self.get_depth_loss(gt_depth, depth)
        return losses

    def simple_test(self, points, img_metas, img=None, rescale=False, **kwargs):
        feats, _, _ = self.extract_feat(points, img, img_metas, **kwargs)
        boxes = self.simple_test_pts(feats, img_metas, rescale=rescale)
        occ = self.occ_head.get_occ(self.occ_head(self.occ_adapter(feats[0])))
        return [dict(pts_bbox=box, occ=labels) for box, labels in zip(boxes, occ)]
