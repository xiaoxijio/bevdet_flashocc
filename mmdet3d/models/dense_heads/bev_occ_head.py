# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from Yzichen/FlashOCC, projects/mmdet3d_plugin/models/dense_heads/bev_occ_head.py.
# Apache-2.0; see third_party/flashocc/LICENSE and README.md for local changes.
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from ..builder import HEADS, build_loss

nusc_class_frequencies = np.array([
    944004,
    1897170,
    152386,
    2391677,
    16957802,
    724139,
    189027,
    2074468,
    413451,
    2384460,
    5916653,
    175883646,
    4275424,
    51393615,
    61411620,
    105975596,
    116424404,
    1892500630
])


@HEADS.register_module()
class BEVOCCHead2D(BaseModule):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_balance=False,
                 loss_occ=None,
                 feature_upsample_scale=1,
                 feature_align_corners=False,
                 use_conv_predicter=False,
                 predictor_hidden_dim=None,
                 ):
        super(BEVOCCHead2D, self).__init__()
        if isinstance(feature_upsample_scale, bool) or not isinstance(feature_upsample_scale, int) or feature_upsample_scale < 1:
            raise ValueError('feature_upsample_scale must be a positive integer')
        if feature_upsample_scale != 1 and not use_predicter:
            raise ValueError('Feature upsampling requires use_predicter=True')
        self.feature_upsample_scale = feature_upsample_scale
        self.feature_align_corners = feature_align_corners
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        hidden_dim = self.out_dim * 2 if predictor_hidden_dim is None else predictor_hidden_dim
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("predictor_hidden_dim must be a positive integer")
        self.predictor_hidden_dim = hidden_dim
        self.use_predicter = use_predicter
        self.use_conv_predicter = use_conv_predicter
        if use_predicter:
            if self.use_conv_predicter:
                self.predicter = nn.Sequential(
                    nn.Conv2d(self.out_dim, hidden_dim, 1),
                    nn.Softplus(),
                    nn.Conv2d(hidden_dim, num_classes * Dz, 1))
            else:
                self.predicter = nn.Sequential(
                    nn.Linear(self.out_dim, hidden_dim),
                    nn.Softplus(),
                    nn.Linear(hidden_dim, num_classes * Dz))

        self.use_mask = use_mask
        self.num_classes = num_classes

        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.register_buffer('cls_weights', class_weights)
            loss_occ['class_weight'] = class_weights        # ce loss
        self.loss_occ = build_loss(loss_occ)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        # Old FP Linear checkpoints are equivalent to pointwise convolutions.
        if self.use_conv_predicter and self.use_predicter:
            for index in (0, 2):
                key = prefix + f'predicter.{index}.weight'
                if key in state_dict and state_dict[key].ndim == 2:
                    state_dict[key] = state_dict[key][:, :, None, None]
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                     strict, missing_keys, unexpected_keys, error_msgs)

    def forward_nchw(self, img_feats):
        """Shared training/export path; channel = z * num_classes + class."""
        features = self.final_conv(img_feats)
        if self.feature_upsample_scale != 1:
            features = F.interpolate(
                features, scale_factor=self.feature_upsample_scale,
                mode='bilinear', align_corners=self.feature_align_corners)
        if self.use_predicter:
            if not self.use_conv_predicter:
                raise ValueError('forward_nchw requires use_conv_predicter=True')
            features = self.predicter(features)
        return features

    def forward(self, img_feats):
        if self.use_conv_predicter or not self.use_predicter:
            raw = self.forward_nchw(img_feats)
            bs, _, dy, dx = raw.shape
            return raw.reshape(bs, self.Dz, self.num_classes, dy, dx).permute(0, 4, 3, 1, 2)
        # Preserve the existing FP experiments and their checkpoint layout.
        features = self.final_conv(img_feats)
        if self.feature_upsample_scale != 1:
            features = F.interpolate(
                features, scale_factor=self.feature_upsample_scale,
                mode='bilinear', align_corners=self.feature_align_corners)
        occ_pred = self.predicter(features.permute(0, 3, 2, 1))
        bs, dx, dy = occ_pred.shape[:3]
        return occ_pred.reshape(bs, dx, dy, self.Dz, self.num_classes)

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        if occ_pred.shape[:-1] != voxel_semantics.shape or mask_camera.shape != voxel_semantics.shape:
            raise ValueError('Occupancy logits, labels and masks must share XYZ axes')
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            voxel_semantics = voxel_semantics.reshape(-1)
            # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
            preds = occ_pred.reshape(-1, self.num_classes)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            mask_camera = mask_camera.reshape(-1)

            if self.class_balance:
                valid_voxels = voxel_semantics[mask_camera.bool()]
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (valid_voxels == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = (mask_camera.bool() & (voxel_semantics != 255)).sum().clamp_min(1)

            loss_occ = self.loss_occ(
                preds,      # (B*Dx*Dy*Dz, n_cls)
                voxel_semantics,    # (B*Dx*Dy*Dz, )
                mask_camera,        # (B*Dx*Dy*Dz, )
                avg_factor=torch.as_tensor(num_total_samples, device=preds.device).clamp_min(1)
            )
            loss['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)

            if self.class_balance:
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (voxel_semantics == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = (voxel_semantics != 255).sum().clamp_min(1)

            loss_occ = self.loss_occ(
                preds,
                voxel_semantics,
                avg_factor=torch.as_tensor(num_total_samples, device=preds.device).clamp_min(1)
            )

            loss['loss_occ'] = loss_occ
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)


