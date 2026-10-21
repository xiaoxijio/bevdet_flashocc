"""Runtime calibration must change voxel assignment in the same exported graph."""
import sys
from pathlib import Path
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from export_bev_occ_onnx import RuntimeGeometryPool, FixedPool
from mmdet3d.models.necks.view_transformer import LSSViewTransformer


def fixture():
    vt = LSSViewTransformer(grid_config=dict(x=[-3, 3, 1], y=[-3, 3, 1],
        z=[-1, 5, 6], depth=[1, 3, 1]), input_size=(2, 3), downsample=1,
        in_channels=4, out_channels=3)
    pool = RuntimeGeometryPool(vt.frustum, vt.grid_lower_bound, vt.grid_interval,
                               vt.grid_size, 3).eval()
    sensors = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
    sensors[:, :, 0, 3] = -.7
    intrinsic = torch.tensor([[2., .1, .3], [0., 2., .2], [0., 0., 1.]])[None, None].repeat(1, 2, 1, 1)
    post_rot = torch.tensor([[.9, -.1, 0.], [.1, .9, 0.], [0., 0., 1.]])[None, None].repeat(1, 2, 1, 1)
    post_trans = torch.tensor([.1, .2, 0.])[None, None].repeat(1, 2, 1)
    bda = torch.eye(4)[None]
    bda[0, 1, 3] = .1
    torch.manual_seed(24)
    depth = torch.rand(2, 2, 2, 3)
    feat = torch.rand(2, 2, 3, 3)
    return vt, pool, (depth, feat, sensors, torch.inverse(intrinsic), torch.inverse(post_rot), post_trans, bda)


def native_reference(vt, inputs):
    depth, feat, sensor, intrinsic, post_rot, post_trans, bda = inputs
    coordinates = vt.get_lidar_coor(sensor, None, torch.inverse(intrinsic), torch.inverse(post_rot), post_trans, bda)
    metadata = vt.voxel_pooling_prepare_v2(coordinates)
    assert metadata[0] is not None
    return FixedPool(metadata, vt.grid_size, 3)(depth, feat)


def test_pool_matches_native_geometry_and_duplicate_sums():
    vt, pool, inputs = fixture()
    _, _, sensor, intrinsic, post_rot, post_trans, bda = inputs
    torch.testing.assert_close(pool.coordinates(*inputs[2:]),
        vt.get_lidar_coor(sensor, None, torch.inverse(intrinsic), torch.inverse(post_rot), post_trans, bda).reshape(-1, 3), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(pool(*inputs), native_reference(vt, inputs), rtol=1e-5, atol=1e-6)
    moved = list(inputs)
    moved[2] = sensor.clone()
    moved[2][:, :, 0, 3] += 1.3
    torch.testing.assert_close(pool(*moved), native_reference(vt, moved), rtol=1e-5, atol=1e-6)
    assert not torch.allclose(pool(*inputs), pool(*moved))


def test_same_onnx_handles_changed_calibration_and_empty_projection(tmp_path):
    vt, pool, inputs = fixture()
    names = ['depth', 'feat', 'sensor2keyegos', 'intrins_inv', 'post_rots_inv', 'post_trans', 'bda']
    path = tmp_path / 'runtime_geometry.onnx'
    torch.onnx.export(pool, inputs, str(path), opset_version=16,
                      input_names=names, output_names=['bev'])
    graph = onnx.shape_inference.infer_shapes(onnx.load(str(path)), strict_mode=True)
    onnx.checker.check_model(graph, full_check=True)
    assert [v.name for v in graph.graph.input] == names
    assert all(not node.domain for node in graph.graph.node)
    assert not any(node.op_type in ('MatMul', 'Gemm') for node in graph.graph.node)
    # Protect against the high-rank matrix broadcasts rejected during OMG build.
    shapes = list(graph.graph.value_info) + list(graph.graph.input) + list(graph.graph.output)
    assert max(len(v.type.tensor_type.shape.dim) for v in shapes) <= 4
    scatter = [node for node in graph.graph.node if node.op_type == 'ScatterND']
    assert len(scatter) == 1
    assert next(a.s for a in scatter[0].attribute if a.name == 'reduction') == b'add'
    session = ReferenceEvaluator(graph)
    for translation in (0., 1.3, 100.):
        runtime = [x.clone() for x in inputs]
        runtime[2][:, :, 0, 3] += translation
        actual = session.run(None, {name: x.numpy() for name, x in zip(names, runtime)})[0]
        np.testing.assert_allclose(actual, pool(*runtime).numpy(), rtol=1e-5, atol=1e-6)
        if translation == 100.:
            assert np.count_nonzero(actual) == 0
        else:
            np.testing.assert_allclose(actual, native_reference(vt, runtime).numpy(), rtol=1e-5, atol=1e-6)
