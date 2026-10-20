"""Regression coverage for opt-in EP1 neck/head/backbone changes."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from mmdet3d.models.necks.lss_fpn import FPN_LSS
from mmdet3d.models.dense_heads.bev_occ_head import BEVOCCHead2D
from mmdet3d.models.backbones.efficientnet_b0_nose import EfficientNetB0NoSE
from torchvision.ops.misc import SqueezeExcitation


def test_neck_legacy_defaults_preserve_weights_and_numerics():
    old = FPN_LSS(20, 8).eval()
    explicit = FPN_LSS(20, 8, channels_factor=2, align_corners=True).eval()
    explicit.load_state_dict(old.state_dict(), strict=True)
    features = [torch.randn(1, 4, 8, 12), None, torch.randn(1, 16, 2, 3)]
    with torch.no_grad():
        torch.testing.assert_close(old(features), explicit(features), rtol=0, atol=0)
    assert old.conv[0].out_channels == 16
    assert FPN_LSS(20, 8, extra_upsample=None).conv[0].out_channels == 8


def test_neck_reduces_internal_channels_and_changes_both_resizes():
    neck = FPN_LSS(20, 8, channels_factor=1, align_corners=False).eval()
    assert neck.conv[0].out_channels == 8
    assert neck.up.align_corners is False
    assert neck.up2[0].align_corners is False
    x = neck([torch.randn(1, 4, 8, 12), None, torch.randn(1, 16, 2, 3)])
    assert x.shape == (1, 8, 16, 24)
    x.sum().backward()
    assert neck.conv[0].weight.grad is not None
    with pytest.raises(ValueError):
        FPN_LSS(20, 8, channels_factor=0)


def test_coarse_head_restores_xy_order_and_supports_original_loss():
    head = BEVOCCHead2D(in_dim=4, out_dim=8, Dz=2, num_classes=18,
        feature_upsample_scale=2, loss_occ=dict(type='CrossEntropyLoss',
            use_sigmoid=False, ignore_index=255))
    x = torch.randn(1, 4, 3, 5, requires_grad=True)
    logits = head(x)
    assert logits.shape == (1, 10, 6, 2, 18)
    reference = F.interpolate(head.final_conv(x), scale_factor=2,
                              mode='bilinear', align_corners=False)
    reference = head.predicter(reference.permute(0, 3, 2, 1)).reshape(1, 10, 6, 2, 18)
    torch.testing.assert_close(logits, reference)
    labels = torch.randint(18, (1, 10, 6, 2))
    mask = torch.ones_like(labels, dtype=torch.bool)
    loss = head.loss(logits, labels, mask)['loss_occ']
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad.isfinite().all() and torch.count_nonzero(x.grad) > 0
    assert head.get_occ(logits.detach())[0].shape == (10, 6, 2)


def test_coarse_head_requires_feature_predictor():
    with pytest.raises(ValueError):
        BEVOCCHead2D(use_predicter=False, feature_upsample_scale=2)


def test_nose_backbone_output_contract_and_pretrained_loading(tmp_path):
    model = EfficientNetB0NoSE().eval()
    assert not any(isinstance(m, (SqueezeExcitation, nn.SiLU)) for m in model.modules())
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 256, 704))
    assert [tuple(t.shape) for t in outputs] == [(1, 112, 16, 44), (1, 1280, 8, 22)]
    path = tmp_path / 'features.pth'
    torch.save(model.state_dict(), path)
    restored = EfficientNetB0NoSE(pretrained=str(path))
    restored.init_weights()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
    broken = tmp_path / 'wrong.pth'
    torch.save({}, broken)
    with pytest.raises(ValueError):
        EfficientNetB0NoSE(pretrained=str(broken)).init_weights()
