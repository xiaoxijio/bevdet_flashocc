"""Crop Occ3D labels/masks on the original 0.4 m grid, after augmentation."""
import numpy as np
from ..builder import PIPELINES


def validate_occ_crop(bounds):
    if (len(bounds) != 2 or any(isinstance(x, bool) or not isinstance(x, int) for x in bounds)
            or not 0 <= bounds[0] < bounds[1] <= 200):
        raise ValueError('bounds must be two integer cell edges within [0, 200]')


@PIPELINES.register_module()
class CropOccTargets:
    def __init__(self, bounds=(36, 164)):
        validate_occ_crop(bounds)
        self.bounds = tuple(bounds)

    def __call__(self, results):
        start, end = self.bounds
        for key in ('voxel_semantics', 'mask_lidar', 'mask_camera'):
            value = results[key]
            if value.shape[:2] != (200, 200):
                raise ValueError('CropOccTargets expects original 200x200 Occ3D labels')
            results[key] = np.ascontiguousarray(value[start:end, start:end, ...])
        return results
