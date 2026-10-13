# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.parallel import DataContainer
from ..builder import PIPELINES
from .loading import BEVAug


@PIPELINES.register_module()
class BEVAugDetOcc(BEVAug):
    """Use the existing box/voxel flips; reject unsupported voxel transforms."""
    def __init__(self, bda_aug_conf, classes, is_train=True):
        if (tuple(bda_aug_conf['rot_lim']) != (0., 0.) or
                tuple(bda_aug_conf['scale_lim']) != (1., 1.) or
                any(bda_aug_conf.get('tran_lim', (0., 0., 0.)))):
            raise ValueError('Joint voxel augmentation only supports x/y flips')
        super().__init__(bda_aug_conf, classes, is_train)


@PIPELINES.register_module()
class FormatOccTargets:
    def __call__(self, results):
        for key in ('voxel_semantics', 'mask_lidar', 'mask_camera'):
            dtype = torch.long if key == 'voxel_semantics' else torch.bool
            tensor = torch.as_tensor(np.ascontiguousarray(results[key]), dtype=dtype)
            results[key] = DataContainer(tensor, stack=True, pad_dims=None)
        return results
