import torch
import torch.nn.functional as F
from mmdet3d.models.detectors.bevdet_det_occ import BEVDetDetOcc


def test_matches_probability_bce_loss_and_gradient():
    torch.manual_seed(7)
    logits = torch.randn(11, 59, dtype=torch.double, requires_grad=True)
    labels = F.one_hot(torch.arange(11) % 59, 59).double()
    expected = F.binary_cross_entropy(logits.softmax(-1), labels, reduction='sum')
    actual = BEVDetDetOcc.depth_bce_with_logits(logits, labels)
    expected_grad, = torch.autograd.grad(expected, logits, retain_graph=True)
    actual_grad, = torch.autograd.grad(actual, logits)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_saturated_wrong_prediction_has_finite_corrective_gradient():
    for scale in (100., 1.e6):
        logits = torch.tensor([[scale, -scale, 0.]], requires_grad=True)
        labels = torch.tensor([[0., 1., 0.]])
        loss = BEVDetDetOcc.depth_bce_with_logits(logits, labels)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(logits.grad).all()
        assert logits.grad[0, 0] > 0
        assert logits.grad[0, 1] < 0


def test_empty_depth_mask_returns_differentiable_zero():
    from types import SimpleNamespace
    from mmdet3d.models.necks.view_transformer import LSSViewTransformer
    vt = LSSViewTransformer(
        grid_config=dict(x=[0, 2, 1], y=[0, 2, 1], z=[0, 1, 1], depth=[1, 4, 1]),
        input_size=(2, 2), downsample=1, in_channels=2, out_channels=2)
    model = SimpleNamespace(img_view_transformer=vt,
                            depth_loss_weight=0.1, depth_loss_from_logits=True)
    logits = torch.randn(1, 3, 2, 2, requires_grad=True)
    loss = BEVDetDetOcc.get_depth_loss(model, torch.zeros(1, 1, 2, 2), logits)
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_logits_option_preserves_pooling_and_default_probability_output():
    from mmdet3d.models.necks.view_transformer import LSSViewTransformer
    vt = LSSViewTransformer(
        grid_config=dict(x=[0, 2, 1], y=[0, 2, 1], z=[0, 1, 1], depth=[1, 4, 1]),
        input_size=(2, 2), downsample=1, in_channels=2, out_channels=2)
    pooled_depth = []
    def pool(inputs, depth, context):
        pooled_depth.append(depth.detach().clone())
        return context, depth
    vt.view_transform = pool
    inputs = [torch.randn(1, 1, 2, 2, 2)]
    bev, probabilities = vt(inputs)
    bev_logits, logits = vt(inputs, return_depth_logits=True)
    torch.testing.assert_close(probabilities, logits.softmax(1))
    torch.testing.assert_close(bev, bev_logits)
    torch.testing.assert_close(pooled_depth[0], pooled_depth[1])
