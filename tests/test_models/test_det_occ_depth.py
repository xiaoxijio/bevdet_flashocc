from types import SimpleNamespace
import torch
from mmcv import Config
from mmdet3d.models.detectors.bevdet_det_occ import BEVDetDetOcc
from mmdet3d.models.necks.view_transformer import LSSViewTransformerBEVDepth


def test_depth_loss_matches_bevdepth_and_backpropagates():
    vt = LSSViewTransformerBEVDepth(
        grid_config=dict(x=[-2,2,1], y=[-2,2,1], z=[-1,1,2], depth=[1,5,1]),
        input_size=(4,4), downsample=2, in_channels=4, out_channels=4,
        loss_depth_weight=1.)
    model = SimpleNamespace(img_view_transformer=vt, depth_loss_weight=1.)
    gt = torch.zeros(1,1,4,4)
    gt[0,0,0,0] = 2.2
    gt[0,0,1,1] = 3.2
    gt[0,0,2,2] = 4.8
    logits = torch.randn(1,4,2,2,requires_grad=True)
    pred = logits.softmax(1)
    loss = BEVDetDetOcc.get_depth_loss(model, gt, pred)
    torch.testing.assert_close(loss, vt.get_depth_loss(gt,pred))
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    empty = BEVDetDetOcc.get_depth_loss(model, torch.zeros_like(gt), pred)
    assert empty.item() == 0 and torch.isfinite(empty)


def test_depth_enabled_only_for_g2_to_g4():
    from pathlib import Path
    for p in Path('configs/bevdet/ep1_bevdet+flashocc').glob('*.py'):
        c = Config.fromfile(str(p))
        enabled = 'g1-' not in p.name
        assert c.model.get('depth_loss_weight',0) == (1. if enabled else 0.)
        pipe = c.data.train.dataset.pipeline
        assert ('gt_depth' in pipe[-1]['keys']) == enabled
        assert sum(x['type']=='PointToMultiViewDepth' for x in pipe) == int(enabled)
        assert c.model.img_view_transformer.type == 'LSSViewTransformer'
        assert c.data.samples_per_gpu == 2


def test_forward_train_routes_depth_and_requires_labels():
    import pytest
    from unittest.mock import Mock
    head = Mock(return_value=torch.ones(1))
    head.loss.return_value = {'loss_occ': torch.tensor(2.)}
    model = SimpleNamespace(
        depth_loss_weight=1., occ_loss_weight=1.,
        extract_feat=Mock(return_value=([torch.ones(1)], None, torch.tensor(3.))),
        forward_pts_train=Mock(side_effect=lambda *args: {'loss_det': torch.tensor(4.)}),
        occ_adapter=lambda x: x, occ_head=head,
        get_depth_loss=Mock(return_value=torch.tensor(5.)))
    args = dict(voxel_semantics=torch.ones(1), mask_camera=torch.ones(1))
    with pytest.raises(ValueError, match='requires gt_depth'):
        BEVDetDetOcc.forward_train(model, **args)
    gt = torch.ones(1,1,2,2)
    losses = BEVDetDetOcc.forward_train(model, gt_depth=gt, **args)
    assert set(losses) == {'loss_det','loss_occ','loss_depth'}
    model.get_depth_loss.assert_called_once()
    model.depth_loss_weight = 0.
    losses = BEVDetDetOcc.forward_train(model, **args)
    assert 'loss_depth' not in losses
