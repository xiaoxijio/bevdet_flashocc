import numpy as np
import pytest
import torch
import onnx
from onnx.reference import ReferenceEvaluator
from tools.export_bev_occ_onnx import FixedPool
from mmdet3d.models.necks.view_transformer import LSSViewTransformer
from tools.bev_occ_mapping import make_layout, make_mapping, mapping_names, RuntimeMappingPool


def fixture():
    vt = LSSViewTransformer(grid_config=dict(x=[-3, 3, 1], y=[-3, 3, 1],
        z=[-1, 5, 6], depth=[1, 3, 1]), input_size=(2, 3), downsample=1,
        in_channels=4, out_channels=3)
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
    return vt, None, (depth, feat, sensors, torch.inverse(intrinsic), torch.inverse(post_rot), post_trans, bda)


def native_reference(vt, inputs):
    depth, feat, sensor, intrinsic, post_rot, post_trans, bda = inputs
    coordinates = vt.get_lidar_coor(sensor, None, torch.inverse(intrinsic), torch.inverse(post_rot), post_trans, bda)
    metadata = vt.voxel_pooling_prepare_v2(coordinates)
    assert metadata[0] is not None
    return FixedPool(metadata, vt.grid_size, 3)(depth, feat)



def metadata(vt, inputs):
    _, _, s, k, p, t, b = inputs
    return vt.voxel_pooling_prepare_v2(vt.get_lidar_coor(s, None, torch.inverse(k), torch.inverse(p), t, b))


def test_same_onnx_accepts_changed_and_empty_maps(tmp_path):
    vt, _, inputs = fixture()
    layout = make_layout(metadata(vt, inputs), vt.grid_size, 3, reserve=4)
    pool = RuntimeMappingPool(layout).eval()
    maps = make_mapping(metadata(vt, inputs), layout)
    torch.testing.assert_close(pool(*inputs[:2], *maps), native_reference(vt, inputs))
    path = tmp_path / 'pool.onnx'
    names = ['depth', 'feat'] + mapping_names(layout)
    torch.onnx.export(pool, (*inputs[:2], *maps), str(path), input_names=names, output_names=['bev'], opset_version=16)
    graph = onnx.shape_inference.infer_shapes(onnx.load(path), strict_mode=True)
    onnx.checker.check_model(graph, full_check=True)
    assert not any(n.op_type.startswith('Scatter') for n in graph.graph.node)
    assert [i.name for i in graph.graph.input] == names
    evaluator = ReferenceEvaluator(graph)
    for shift in (0., 1.3, 10000.):
        moved = list(inputs)
        moved[2] = moved[2].clone()
        moved[2][:, :, 0, 3] += shift
        meta = metadata(vt, moved)
        mapping = make_mapping(meta, layout)
        args = (*inputs[:2], *mapping)
        actual = evaluator.run(None, {n: t.numpy() for n,t in zip(names,args)})[0]
        expected = torch.zeros_like(pool(*args)) if meta[0] is None else native_reference(vt,moved)
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-6)


def test_capacity_overflow_is_rejected():
    vt, _, inputs = fixture()
    meta = metadata(vt, inputs)
    layout = make_layout(meta, vt.grid_size, 3)
    for g in layout['groups']:
        g['count'] = 0
    with pytest.raises(ValueError, match='capacity'):
        make_mapping(meta, layout)


def test_encoded_large_indices_survive_fp16():
    from tools.bev_occ_mapping import encode_indices, decode_indices
    indices = np.array([0, 1, 1023, 1024, 16383, 249215, 1048575])
    encoded = torch.from_numpy(encode_indices(indices)).half().float()
    np.testing.assert_array_equal(decode_indices(encoded).numpy(), indices)
    with pytest.raises(ValueError, match='capacity'):
        encode_indices(np.array([1048576]))
