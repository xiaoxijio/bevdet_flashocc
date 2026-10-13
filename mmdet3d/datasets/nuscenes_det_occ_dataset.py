# Copyright (c) OpenMMLab. All rights reserved.
import os
import numpy as np
from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset

OCC_CLASSES = ('others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
               'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
               'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
               'manmade', 'vegetation', 'free')


@DATASETS.register_module()
class NuScenesDatasetDetOcc(NuScenesDataset):
    """Joint output evaluation with camera-visible Occ3D voxels.

    IoU values are percentages; mIoU excludes free (class 17), matching Occ3D.
    """
    def __init__(self, occ_gt_root=None, **kwargs):
        super().__init__(**kwargs)
        self.occ_gt_root = occ_gt_root or os.path.join(self.data_root, 'gts')

    def _occ_path(self, index):
        info = self.data_infos[index]
        path = info.get('occ_path')
        if not path:
            raise KeyError('PKL needs occ_path; run tools/create_data_bevdet.py')
        scene = os.path.basename(os.path.dirname(os.path.normpath(path)))
        return os.path.join(self.occ_gt_root, scene, info['token'], 'labels.npz')

    def get_data_info(self, index):
        result = super().get_data_info(index)
        result['occ_gt_path'] = os.path.dirname(self._occ_path(index))
        return result

    def _evaluate_occ(self, results):
        hist = np.zeros((18, 18), dtype=np.int64)
        for index, result in enumerate(results):
            pred = np.asarray(result['occ'])
            with np.load(self._occ_path(index)) as labels:
                gt = labels['semantics']
                mask = labels['mask_camera'].astype(bool) & (gt < 18)
                if pred.shape != gt.shape:
                    raise ValueError('Prediction and Occ3D GT shapes differ')
                if np.any((pred < 0) | (pred >= 18)):
                    raise ValueError('Occupancy prediction must use labels 0..17')
                hist += np.bincount(18 * gt[mask].astype(np.int64) +
                                    pred[mask].astype(np.int64), minlength=324).reshape(18, 18)
        union = hist.sum(0) + hist.sum(1) - np.diag(hist)
        iou = np.divide(np.diag(hist), union, out=np.full(18, np.nan), where=union > 0)
        metrics = {'occ/' + name + '_IoU': float(iou[i] * 100)
                   for i, name in enumerate(OCC_CLASSES)}
        finite = iou[:17][np.isfinite(iou[:17])]
        metrics['occ/mIoU'] = float(finite.mean() * 100) if finite.size else float('nan')
        return metrics

    def evaluate(self, results, metric=('bbox', 'occ'), **kwargs):
        if len(results) != len(self):
            raise ValueError('Expected one result per dataset sample')
        metrics = [metric] if isinstance(metric, str) else list(metric)
        if not set(metrics) <= {'bbox', 'occ'}:
            raise ValueError('Supported metrics: bbox, occ')
        output = {}
        if 'bbox' in metrics:
            # Do not let the base formatter mistake occ for another box head.
            det_results = [dict(pts_bbox=item['pts_bbox']) for item in results]
            output.update(super().evaluate(det_results, metric='bbox', **kwargs))
        if 'occ' in metrics:
            output.update(self._evaluate_occ(results))
        return output

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        det_results = [dict(pts_bbox=item['pts_bbox']) for item in results]
        return super().format_results(det_results, jsonfile_prefix, **kwargs)
