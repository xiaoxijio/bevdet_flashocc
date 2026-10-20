"""Equivalent OCC logits/axes and a transpose-free, executable ONNX branch."""
import sys
from pathlib import Path
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from export_bev_occ_onnx import ExportOccHeadNCHW
from mmdet3d.models.dense_heads.bev_occ_head import BEVOCCHead2D


@pytest.mark.parametrize('predictor,scale', [(False, 1), (True, 1), (True, 2)])
def test_nchw_matches_legacy_values_and_labels(predictor, scale):
    torch.manual_seed(32)
    head = BEVOCCHead2D(in_dim=4, out_dim=8, Dz=2, num_classes=3,
        use_predicter=predictor, feature_upsample_scale=scale,
        loss_occ=dict(type='CrossEntropyLoss', use_sigmoid=False)).eval()
    exported = ExportOccHeadNCHW(head).eval()
    # Non-square XY and multiple heights/classes catch otherwise hidden swaps.
    x = torch.randn(2, 4, 3, 5)
    with torch.no_grad():
        legacy, raw = head(x), exported(x)
    assert raw.shape == (2, 6, 3 * scale, 5 * scale)
    restored = raw.reshape(2, 2, 3, 3 * scale, 5 * scale).permute(0, 4, 3, 1, 2)
    torch.testing.assert_close(restored, legacy, rtol=1e-5, atol=1e-6)
    labels = raw.reshape(2, 2, 3, 3 * scale, 5 * scale).argmax(2).permute(0, 3, 2, 1)
    torch.testing.assert_close(labels, legacy.argmax(-1), rtol=0, atol=0)


@pytest.mark.parametrize('scale', [1, 2])
def test_onnx_keeps_nchw_and_matches_runtime(tmp_path, scale):
    torch.manual_seed(42)
    head = BEVOCCHead2D(in_dim=4, out_dim=8, Dz=2, num_classes=3,
        feature_upsample_scale=scale,
        loss_occ=dict(type='CrossEntropyLoss', use_sigmoid=False)).eval()
    exported = ExportOccHeadNCHW(head).eval()
    x = torch.randn(1, 4, 3, 5)
    path = tmp_path / 'occ_nchw.onnx'
    torch.onnx.export(exported, x, str(path), opset_version=16,
                      input_names=['features'], output_names=['occ_logits'])
    graph = onnx.shape_inference.infer_shapes(onnx.load(str(path)), strict_mode=True)
    onnx.checker.check_model(graph, full_check=True)
    types = [node.op_type for node in graph.graph.node]
    assert 'Transpose' not in types
    assert 'MatMul' not in types and 'Gemm' not in types
    assert types.count('Conv') == 3
    assert [d.dim_value for d in graph.graph.output[0].type.tensor_type.shape.dim] == [1, 6, 3 * scale, 5 * scale]
    session = ReferenceEvaluator(graph)
    actual = session.run(None, {'features': x.numpy()})[0]
    with torch.no_grad():
        expected = head(x).numpy()
    restored = actual.reshape(1, 2, 3, 3 * scale, 5 * scale).transpose(0, 4, 3, 1, 2)
    np.testing.assert_allclose(restored, expected, rtol=1e-5, atol=1e-6)
