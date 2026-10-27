"""Restore the SDK optimized detector's trained function, then attach G3 OCC.

The SDK file is an inference export, not the unpublished original training PTH.
Conv/BN folding is inverted non-uniquely using an identity running transform.
Frozen BN statistics are therefore mandatory for the corresponding G5 recipe.
Every detection convolution is checked and numerical frontend/backend parity is
required before the initialization checkpoint is written.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import torch
from torch import nn
from mmcv import Config
from mmdet3d.models import build_model


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def paired_bn(name, modules):
    prefix, leaf = name.rsplit('.', 1)
    parent = modules[prefix]
    candidate = None
    if hasattr(parent, 'expansion') and leaf in ('conv_pw', 'conv_dw', 'conv_pwl'):
        if parent.expansion == 1:
            candidate = parent.bn1 if leaf == 'conv_dw' else parent.bn2
        else:
            candidate = {'conv_pw': parent.bn1, 'conv_dw': parent.bn2,
                         'conv_pwl': parent.bn3}[leaf]
    elif leaf in ('conv_stem', 'conv_head'):
        candidate = parent.bn1 if leaf == 'conv_stem' else parent.bn2
    elif leaf in ('conv1', 'conv2', 'conv3'):
        candidate = getattr(parent, 'bn' + leaf[-1], None)
    elif leaf == 'conv':
        candidate = getattr(parent, 'norm', None)
    elif leaf.isdigit() and isinstance(parent, nn.Sequential):
        index = int(leaf) + 1
        candidate = parent[index] if index < len(parent) else None
    if isinstance(candidate, nn.modules.batchnorm._BatchNorm):
        return next((k, m) for k, m in modules.items() if m is candidate)
    return None, None


def copy_official(model, graph):
    initializers = {x.name: numpy_helper.to_array(x).copy() for x in graph.graph.initializer}
    source = [n for n in graph.graph.node if n.op_type == 'Conv']
    modules = dict(model.named_modules())
    target = [(name, module) for name, module in modules.items()
              if isinstance(module, nn.Conv2d) and not name.startswith(('occ_adapter.', 'occ_head.'))]
    execution_order = ['img_backbone.', 'img_neck.', 'img_view_transformer.',
                       'img_bev_encoder_backbone.', 'img_bev_encoder_neck.', 'pts_bbox_head.']
    target.sort(key=lambda item: next(i for i, prefix in enumerate(execution_order)
                                     if item[0].startswith(prefix)))
    if len(source) != len(target):
        raise ValueError(f'Conv count mismatch: SDK {len(source)}, G5 {len(target)}')
    report = []
    assigned_bn = set()
    with torch.no_grad():
        for node, (name, conv) in zip(source, target):
            w = initializers[node.input[1]]
            bias = initializers[node.input[2]] if len(node.input) > 2 else np.zeros(w.shape[0], dtype=np.float32)
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            expected = dict(group=conv.groups, strides=list(conv.stride),
                            dilations=list(conv.dilation),
                            pads=list(conv.padding) + list(conv.padding))
            for key, value in expected.items():
                default = 1 if key == 'group' else ([0]*4 if key == 'pads' else [1, 1])
                if attrs.get(key, default) != value:
                    raise ValueError(f'{node.name} -> {name}: {key} mismatch')
            if tuple(w.shape) != tuple(conv.weight.shape):
                raise ValueError(f'{node.name} -> {name}: weight shape mismatch {w.shape} vs {tuple(conv.weight.shape)}')
            conv.weight.copy_(torch.from_numpy(w))
            bn_name, bn = paired_bn(name, modules)
            if conv.bias is not None:
                conv.bias.copy_(torch.from_numpy(bias))
            elif bn is None and np.any(bias):
                raise ValueError(f'Cannot represent fused bias for {name}')
            if bn is not None:
                # eval: (x-0)/sqrt((1-eps)+eps)*1 + folded_bias.
                bn.running_mean.zero_()
                bn.running_var.fill_(1. - bn.eps)
                bn.weight.fill_(1.)
                bn.bias.copy_(torch.from_numpy(bias) if conv.bias is None else torch.zeros_like(bn.bias))
                bn.num_batches_tracked.zero_()
                assigned_bn.add(bn_name)
            report.append(dict(onnx=node.name, pytorch=name, bn=bn_name, weight_shape=list(w.shape)))
    expected_bn = {n for n, m in modules.items() if isinstance(m, nn.modules.batchnorm._BatchNorm)
                   and not n.startswith(('occ_adapter.', 'occ_head.'))}
    if assigned_bn != expected_bn:
        raise ValueError('Unmapped BatchNorm: ' + str(expected_bn - assigned_bn))
    return report


def make_subgraph(model, inputs, outputs):
    """Extract dependencies and topologically sort; legacy SDK graph is unsorted."""
    producers = {v: n for n in model.graph.node for v in n.output}
    initializers = {v.name: v for v in model.graph.initializer}
    selected, visiting, ordered, used_init = set(), set(), [], set()
    def visit(value):
        if not value or value in inputs:
            return
        if value in initializers:
            used_init.add(value)
            return
        if value not in producers:
            raise ValueError('Unresolved ONNX dependency: ' + value)
        node = producers[value]
        key = node.name
        if key in selected:
            return
        if key in visiting:
            raise ValueError('ONNX dependency cycle')
        visiting.add(key)
        for arg in node.input:
            visit(arg)
        visiting.remove(key)
        selected.add(key)
        ordered.append(node)
    for value in outputs:
        visit(value)
    graph = helper.make_graph(ordered, 'g5_sdk_parity',
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in inputs.items()],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in outputs.items()],
        [v for n, v in initializers.items() if n in used_init])
    result = helper.make_model(graph, opset_imports=list(model.opset_import), producer_name='g5-converter')
    result.ir_version = model.ir_version
    onnx.checker.check_model(result)
    return result


def validate(model, source, directory):
    import onnxruntime as ort
    # These boundaries are checked against the supplied SDK model version.
    front = make_subgraph(source, {'img': [6, 3, 256, 704]}, {
        'input.131': [6, 112, 16, 44], 'onnx::Conv_682': [6, 1280, 8, 22],
        'onnx::Shape_703': [6, 256, 16, 44], 'onnx::Slice_751': [6, 123, 16, 44]})
    names = [o.name for o in source.graph.output]
    shapes = [[1, c, 128, 128] for c in [2, 1, 3, 2, 2, 10]]
    back = make_subgraph(source, {'onnx::Conv_919': [1, 64, 128, 128]}, dict(zip(names, shapes)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    sessions = [ort.InferenceSession(g.SerializeToString(), options, providers=['CPUExecutionProvider'])
                for g in (front, back)]
    torch.manual_seed(5)
    records = []
    model.cpu().eval()
    # Two image/BEV inputs test graph equivalence, not only parameter shapes.
    with torch.no_grad():
        for seed in [5, 17]:
            torch.manual_seed(seed)
            image = torch.randn(6, 3, 256, 704)
            features = model.img_backbone(image)
            neck = model.img_neck(features)
            if isinstance(neck, (list, tuple)):
                neck = neck[0]
            logits = model.img_view_transformer.depth_net(neck)
            actual_front = (*features, neck, logits)
            expected_front = sessions[0].run(None, {'img': image.numpy()})
            bev = torch.randn(1, 64, 128, 128)
            encoded = model.bev_encoder(bev)
            heads = model.pts_bbox_head([encoded])[0][0]
            actual_back = [heads[k] for k in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']]
            expected_back = sessions[1].run(None, {'onnx::Conv_919': bev.numpy()})
            for name, expected, actual in zip(['backbone_112', 'backbone_1280', 'image_fpn', 'depth_context_logits'] + names,
                                              expected_front + expected_back, list(actual_front) + actual_back):
                actual = actual.numpy()
                np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=2e-3,
                                           err_msg=f'SDK numerical parity failed: {name}, seed {seed}')
                records.append(dict(seed=seed, tensor=name,
                                    max_abs_error=float(np.max(np.abs(actual - expected)))))
    onnx.save(front, str(directory / 'sdk_frontend_reference.onnx'))
    onnx.save(back, str(directory / 'sdk_backend_reference.onnx'))
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-onnx', required=True, type=Path)
    p.add_argument('--occ-checkpoint', required=True, type=Path)
    p.add_argument('--config', default='configs/bevdet/ep1_bevdet+flashocc/bevdet-b0-det-occ-g5-official.py')
    p.add_argument('--output', default='ckpt/g5_official_det_g3_occ_init.pth', type=Path)
    p.add_argument('--report-dir', default='work_dirs/g5_official_conversion', type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists; choose another --output')
    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model)
    source = onnx.load(str(args.source_onnx))
    mapping = copy_official(model, source)
    ckpt = torch.load(str(args.occ_checkpoint), map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    target = model.state_dict()
    occ_names = [k for k in target if k.startswith(('occ_adapter.', 'occ_head.'))]
    for k in occ_names:
        if k not in state or state[k].shape != target[k].shape:
            raise ValueError('G3 OCC tensor mismatch: ' + k)
        target[k] = state[k]
    model.load_state_dict(target, strict=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    parity = validate(model, source, args.report_dir)
    report = dict(source_onnx=str(args.source_onnx), source_sha256=digest(args.source_onnx),
        occ_checkpoint=str(args.occ_checkpoint), occ_sha256=digest(args.occ_checkpoint),
        detection_convolutions=len(mapping), occ_tensors=len(occ_names), mapping=mapping,
        parity=parity, bn_policy='Equivalent folded affine initialization; frozen running statistics',
        limitations='Not the unpublished original SDK training PTH or joint-training recipe; pooling geometry is not imported')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), meta=dict(g5_official=report)), str(args.output))
    report['output_sha256'] = digest(args.output)
    (args.report_dir / 'conversion.json').write_text(json.dumps(report, indent=2))
    print('Saved:', args.output)
    print('Official detection convs:', len(mapping), 'G3 OCC tensors:', len(occ_names))
    print('Numerical parity checks:', len(parity), 'PASSED')


if __name__ == '__main__':
    main()
