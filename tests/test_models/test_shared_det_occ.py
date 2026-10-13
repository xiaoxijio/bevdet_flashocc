"""Focused correctness tests for joint geometry, supervision and output routing."""
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from mmdet3d.models.necks.bev_grid_adapter import BEVGridAdapter
from mmdet3d.models.dense_heads.bev_occ_head import BEVOCCHead2D
from mmdet3d.datasets.nuscenes_det_occ_dataset import NuScenesDatasetDetOcc
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from mmdet3d.datasets.pipelines.det_occ import BEVAugDetOcc


def test_adapter_metric_centers_and_xy_order():
    adapter = BEVGridAdapter([-2., 2., 1.], [-1., 1., .5], 1, 1)
    with torch.no_grad():
        adapter.proj[0].weight.fill_(1.)
        adapter.proj[0].bias.zero_()
    x = torch.arange(4.).view(1, 1, 1, 4).expand(1, 1, 4, 4)
    actual = adapter(x)
    expected = torch.tensor([.75, 1.25, 1.75, 2.25])
    torch.testing.assert_close(actual[0, 0, 0], expected)
    y = x.transpose(2, 3)
    torch.testing.assert_close(adapter(y)[0, 0, :, 0], expected)


def test_official_head_height_axis_and_zero_visible_loss():
    for predictor in (False, True):
        head = BEVOCCHead2D(in_dim=4, out_dim=8, Dz=2, num_classes=18,
            use_predicter=predictor, loss_occ=dict(type='CrossEntropyLoss',
                use_sigmoid=False, ignore_index=255))
        features = torch.randn(1, 4, 3, 5, requires_grad=True)
        logits = head(features)
        assert logits.shape == (1, 5, 3, 2, 18)
        labels = torch.zeros(1, 5, 3, 2, dtype=torch.long)
        mask = torch.zeros_like(labels, dtype=torch.bool)
        loss = head.loss(logits, labels, mask)['loss_occ']
        assert torch.isfinite(loss) and loss.item() == 0.
        loss.backward()
        assert torch.count_nonzero(features.grad) == 0
        assert head.get_occ(logits.detach())[0].shape == (5, 3, 2)


def test_joint_eval_excludes_free_and_strips_occ_from_detection():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / 'scene-0001' / 'sample'
        path.mkdir(parents=True)
        gt = np.array([[[4, 17]]], dtype=np.uint8)
        np.savez(path / 'labels.npz', semantics=gt, mask_camera=np.ones_like(gt),
                 mask_lidar=np.ones_like(gt))
        dataset = NuScenesDatasetDetOcc.__new__(NuScenesDatasetDetOcc)
        dataset.occ_gt_root = str(root)
        dataset.data_infos = [dict(token='sample', occ_path='gts/scene-0001/sample')]
        result = [dict(pts_bbox={'sentinel': True}, occ=np.array([[[4, 4]]]))]
        with patch.object(NuScenesDataset, 'evaluate', return_value={'det': 1.}) as evaluate:
            metrics = dataset.evaluate(result, metric=['bbox', 'occ'])
            assert set(evaluate.call_args.args[0][0]) == {'pts_bbox'}
        assert metrics['occ/car_IoU'] == 50.
        assert metrics['occ/free_IoU'] == 0.
        assert metrics['occ/mIoU'] == 50.


def test_joint_augmentation_flips_xyz_labels_and_masks():
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    aug = BEVAugDetOcc(dict(rot_lim=(0., 0.), scale_lim=(1., 1.),
                           flip_dx_ratio=1., flip_dy_ratio=0.), classes=[])
    voxels = np.arange(12).reshape(3, 2, 2)
    results = dict(gt_bboxes_3d=LiDARInstance3DBoxes(torch.tensor(
        [[1., 2., 0., 4., 2., 1.5, 0., 0., 0.]]), box_dim=9),
        voxel_semantics=voxels.copy(), mask_camera=voxels.copy(), mask_lidar=voxels.copy())
    output = aug(results)
    for key in ('voxel_semantics', 'mask_camera', 'mask_lidar'):
        np.testing.assert_array_equal(output[key], voxels[::-1])
    assert output['gt_bboxes_3d'].tensor[0, 0].item() == -1.
