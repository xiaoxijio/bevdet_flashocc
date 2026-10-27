from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from mmcv import Config
from mmdet3d.models.backbones.efficientnet_b0_hz import (
    EfficientNetB0HZ, G5FoldedBatchNormHook)


def test_sdk_backbone_output_stages_and_activation():
    m = EfficientNetB0HZ().eval()
    with torch.no_grad():
        a, b = m(torch.randn(1, 3, 64, 96))
    assert a.shape == (1, 112, 4, 6)
    assert b.shape == (1, 1280, 2, 3)
    assert not any(isinstance(x, nn.SiLU) for x in m.modules())
    assert sum(isinstance(x, nn.Conv2d) for x in m.modules()) == 49


def test_folded_bn_hook_preserves_stats_and_affine_gradients():
    m = nn.Sequential(nn.Conv2d(2, 3, 1), nn.BatchNorm2d(3))
    m.train()
    before = m[1].running_mean.clone()
    G5FoldedBatchNormHook().before_train_iter(SimpleNamespace(model=m))
    assert not m[1].training and m[1].weight.requires_grad
    m(torch.randn(2, 2, 3, 3)).sum().backward()
    torch.testing.assert_close(before, m[1].running_mean)
    assert m[1].weight.grad is not None


def test_g5_config_is_independent_of_g2_and_g4_changes():
    c = Config.fromfile('configs/bevdet/ep1_bevdet+flashocc/bevdet-b0-det-occ-g5-official.py')
    assert c.model.img_backbone.type == 'EfficientNetB0HZ'
    assert c.model.img_backbone.pretrained is None
    assert c.model.img_neck.in_channels == [112, 1280]
    assert c.model.img_bev_encoder_neck.out_channels == 128
    assert c.model.img_bev_encoder_neck.channels_factor == 1
    assert c.model.occ_adapter.dst_grid == [-40., 40., .4]
    assert c.model.occ_head.feature_upsample_scale == 1
    assert c.model.depth_loss_weight == 0
    assert c.model.transfer_from is None
    assert 'PointToMultiViewDepth' not in [x.type for x in c.train_pipeline]
    assert c.load_from.endswith('g5_official_det_g3_occ_init.pth')
    assert c.resume_from is None and not c.auto_resume
    assert c.data.samples_per_gpu * 4 == 8


def test_official_conversion_covers_every_detection_convolution():
    import onnx
    from mmdet3d.models import build_model
    from tools.convert_g5_official_onnx import copy_official
    source = Path('ckpt/g5_official_bevdet_source.onnx')
    if not source.exists():
        pytest.skip('SDK ONNX not distributed in git; supply it under ckpt/')
    c = Config.fromfile('configs/bevdet/ep1_bevdet+flashocc/bevdet-b0-det-occ-g5-official.py')
    m = build_model(c.model)
    graph = onnx.load(str(source))
    report = copy_official(m, graph)
    assert len(report) == 85
    first = next(n for n in graph.graph.node if n.op_type == 'Conv')
    next(a for a in first.attribute if a.name == 'strides').ints[0] = 1
    with pytest.raises(ValueError, match='strides mismatch'):
        copy_official(m, graph)
