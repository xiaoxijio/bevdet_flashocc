"""Export joint Det/OCC with fixed geometry or cached maps; NCHW OCC logits.

Fixed geometry uses composed indices, zero-depth padding and power-of-two groups.
"""
import argparse
import json
import hashlib
import os
import sys
import warnings
from pathlib import Path
sys.path.insert(0, os.getcwd())
import numpy as np
import onnx
import torch
from torch.nn import functional as F
from mmcv import Config
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model

from tools.bev_occ_mapping import RuntimeMappingPool, make_layout, make_mapping, mapping_names

KEYS = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']

class FixedPool(torch.nn.Module):
    """Compose constant indices before export; never materialize all point features.

    Each bucket gathers depth/features directly in reduction order. Invalid slots
    reference an exact zero depth prefix, replacing the legacy mask multiply.
    """
    def __init__(self, metas, grid_size, channels):
        super().__init__()
        rb, rd, rf, starts, lengths = metas
        if not lengths.numel() or bool((lengths <= 0).any()):
            raise ValueError('Pool needs nonempty positive lengths')
        self.height, self.width = int(grid_size[1]), int(grid_size[0])
        self.channels = channels
        self.groups = []
        ordered_bev = []
        max_length = int(lengths.max().item())
        bounds = [1 << i for i in range((max_length - 1).bit_length() + 1)]
        previous = 0
        for bound in bounds:
            ids = torch.where((lengths > previous) & (lengths <= bound))[0]
            previous = bound
            if not ids.numel():
                continue
            offsets = torch.arange(bound, device=starts.device)[None, :]
            valid = offsets < lengths[ids, None]
            source = torch.where(valid, starts[ids, None] + offsets, 0).long()
            depth_indices = torch.where(valid, rd.long()[source] + 1, 0)
            feature_indices = torch.where(valid, rf.long()[source], 0)
            self.register_buffer(f'rd_{bound}', depth_indices.reshape(-1).long())
            self.register_buffer(f'rf_{bound}', feature_indices.reshape(-1).long())
            self.groups.append((bound, int(ids.numel())))
            ordered_bev.append(rb[starts[ids].long()])
        self.register_buffer('bev_indices', torch.cat(ordered_bev).long())
        self.valid_points = int(lengths.sum().item())
        self.padded_points = sum(bound * count for bound, count in self.groups)

    def forward(self, depth, feat):
        depth = torch.cat((depth.new_zeros((1, 1)), depth.reshape(-1, 1)), dim=0)
        feat = feat.reshape(-1, self.channels)
        groups = []
        for bound, count in self.groups:
            d = depth[getattr(self, f'rd_{bound}')]
            f = feat[getattr(self, f'rf_{bound}')]
            x = d * f
            if count == 1:
                # SDK 2.1.6 mis-infers ReduceSum(axis=1) on [1,K,C].
                # Reduce the existing [K,C] tensor along axis 0 instead.
                groups.append(x.sum(dim=0, keepdim=True).reshape(1, self.channels))
            else:
                x = x.reshape(count, bound, self.channels)
                groups.append(x.sum(dim=1).reshape(count, self.channels))
        bev = feat.new_zeros((self.height * self.width, self.channels))
        bev[self.bev_indices] = torch.cat(groups, dim=0)
        return bev.reshape(1, self.height, self.width, self.channels).permute(0, 3, 1, 2).contiguous()


class ExportOccHeadNCHW(torch.nn.Module):
    """Reuse trained OCC weights while keeping all features/logits in NCHW.

    Output channel z * num_classes + c represents height z and class c.
    To recover legacy logits [B,X,Y,Z,C]:
        raw.reshape(B, Z, C, Y, X).permute(0, 4, 3, 1, 2)
    """
    def __init__(self, head):
        super().__init__()
        self.head = head
        if head.use_predicter and not getattr(head, 'use_conv_predicter', False):
            predictor = head.predicter
            if (not isinstance(predictor, torch.nn.Sequential) or len(predictor) != 3
                    or not isinstance(predictor[0], torch.nn.Linear)
                    or not isinstance(predictor[1], torch.nn.Softplus)
                    or not isinstance(predictor[2], torch.nn.Linear)
                    or predictor[2].out_features != head.Dz * head.num_classes):
                raise ValueError('NCHW export requires Linear/Softplus/Linear OCC predictor')

    def forward(self, features):
        head = self.head
        if getattr(head, 'use_conv_predicter', False):
            return head.forward_nchw(features)
        features = head.final_conv(features)
        scale = getattr(head, 'feature_upsample_scale', 1)
        if scale != 1:
            features = F.interpolate(features, scale_factor=scale, mode='bilinear',
                align_corners=head.feature_align_corners)
        if head.use_predicter:
            first, activation, last = head.predicter
            features = F.conv2d(features, first.weight[:, :, None, None], first.bias)
            features = activation(features)
            features = F.conv2d(features, last.weight[:, :, None, None], last.bias)
        return features


class ExportModel(torch.nn.Module):
    def __init__(self, model, pool):
        super().__init__()
        self.model, self.pool = model, pool
        self.occ_export = ExportOccHeadNCHW(model.occ_head)

    def image_features(self, img):
        m = self.model
        x = m.img_neck(m.img_backbone(img))
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = m.img_view_transformer.depth_net(x)
        d = m.img_view_transformer.D
        depth = x[:, :d].softmax(dim=1)
        feat = x[:, d:d + m.img_view_transformer.out_channels].permute(0, 2, 3, 1)
        return depth, feat

    def decode_bev(self, pooled):
        m = self.model
        bev = m.bev_encoder(pooled)
        det = m.pts_bbox_head([bev])
        return tuple([task[0][key] for task in det for key in KEYS] + [self.occ_export(m.occ_adapter(bev))])
    def forward(self, img):
        # SDK 2.1.6 first calls model(args), then model(*args).
        if isinstance(img, tuple):
            img, = img
        depth, feat = self.image_features(img)
        return self.decode_bev(self.pool(depth, feat))


class RuntimeMappingExportModel(ExportModel):
    def forward(self, img, *mapping):
        if isinstance(img, tuple) and not mapping:
            img, *mapping = img
        depth, feat = self.image_features(img)
        return self.decode_bev(self.pool(depth, feat, *mapping))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # TODO 在此修改路径
    p.add_argument('--config', default='configs/bevdet/ep1_bevdet+flashocc/bevdet-b0-det-occ-g6-occ128-nearfield.py')
    p.add_argument('--checkpoint', default='work_dirs/ep1_g6_occ128_nearfield/epoch_20_ema.pth')
    p.add_argument('--output', help='Defaults to onnx/bev_occ_mapping.onnx in mapping mode, otherwise onnx/bev_occ.onnx')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Device used for export; does not change ONNX inputs')
    p.add_argument('--qat', action='store_true', help='Restore this project QAT checkpoint and use SDK export')
    p.add_argument('--runtime-mapping', action='store_true', help='Use cached runtime pooling maps instead of geometry/scatter')
    p.add_argument('--mapping-layout', help='Shared layout JSON from the original mapping export; required when updating maps')
    p.add_argument('--mapping-only', action='store_true', help='Write input NPZ for new geometry without exporting/recompiling ONNX')
    p.add_argument('--mapping-reserve', type=float, default=1.1, help='Bucket capacity multiplier when creating a shared layout')
    p.add_argument('--sample-index', type=int, default=0, help='Sample used for fixed geometry or example cached maps')
    p.add_argument('--geometry-npz', help='Optional real-rig geometry: sensor2egos/ego2globals/intrins/post_rots/post_trans/bda, batch=1')
    args = p.parse_args()
    if args.output is None:
        args.output = 'onnx/bev_occ_mapping.onnx' if args.runtime_mapping else 'onnx/bev_occ.onnx'
    if args.sample_index < 0:
        p.error('sample-index must be nonnegative')
    if args.mapping_layout and not args.runtime_mapping:
        p.error('--mapping-layout requires --runtime-mapping')
    if args.mapping_only and (not args.runtime_mapping or not args.mapping_layout):
        p.error('--mapping-only requires --runtime-mapping and --mapping-layout')
    device = torch.device(args.device)
    if device.type == 'cuda':
        # PyTorch can use the CUDA runtime even when the optional NVML library
        # is out of sync with the loaded driver. Preserve all other warnings.
        with warnings.catch_warnings(record=True) as startup_warnings:
            warnings.simplefilter('always', UserWarning)
            torch.cuda.init()
            torch.cuda.device_count()
        for warning in startup_warnings:
            if str(warning.message) == "Can't initialize NVML":
                print('Note: NVML is unavailable; CUDA runtime initialized successfully. '
                      'Check NVIDIA driver/library versions separately.')
            else:
                warnings.warn_explicit(warning.message, warning.category,
                                       warning.filename, warning.lineno)
    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None
    cfg.model.img_backbone.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    if args.qat:
        from mmdet3d.utils.oritek_qat import convert_convolutions, validate_recipe
        if not cfg.get('qat', {}).get('enabled', False):
            p.error('--qat requires a QAT-enabled config')
        validate_recipe(checkpoint, cfg.qat)
    else:
        if 'oritek_qat' in checkpoint.get('meta', {}):
            p.error('This is a QAT checkpoint; pass --qat and the matching config')
        model.load_state_dict({k[7:] if k.startswith('module.') else k: v for k, v in state.items()}, strict=True)
    model.to(device).eval()
    vt = model.img_view_transformer
    if int(vt.grid_size[2]) != 1:
        raise ValueError('This exporter supports a single BEV Z bin')
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    if args.sample_index >= len(dataset):
        p.error('sample-index exceeds dataset length')
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(loader)
    for _ in range(args.sample_index):
        next(iterator)
    data = next(iterator)
    inputs = [t.to(device) for t in data['img_inputs'][0]]
    if args.geometry_npz:
        with np.load(args.geometry_npz) as geometry:
            inputs = [inputs[0]] + [torch.from_numpy(geometry[k]).to(device).float() for k in ['sensor2egos', 'ego2globals', 'intrins', 'post_rots', 'post_trans', 'bda']]
    with torch.no_grad():
        prepared = model.prepare_inputs(inputs)
        img = inputs[0].squeeze(0).float().contiguous()
        if args.runtime_mapping:
            coor = vt.get_lidar_coor(*prepared[1:7])
            if not torch.isfinite(coor).all():
                raise ValueError('Mapping geometry produced nonfinite coordinates')
            metas = vt.voxel_pooling_prepare_v2(coor)
            if args.mapping_layout:
                layout = json.loads(Path(args.mapping_layout).read_text())
                if layout.get('version') != 3 or layout.get('padding') != 'zero_depth_prefix':
                    raise ValueError('Expected zero-depth mapping layout version 3; regenerate layout and maps')
                if (layout['height'], layout['width'], layout['channels']) != (int(vt.grid_size[1]), int(vt.grid_size[0]), vt.out_channels):
                    raise ValueError('Mapping layout does not match model grid/channels')
                signature = dict(grid_lower_bound=vt.grid_lower_bound.cpu().tolist(),
                                 grid_interval=vt.grid_interval.cpu().tolist(),
                                 frustum_sha256=hashlib.sha256(vt.frustum.cpu().numpy().tobytes()).hexdigest(), frustum_shape=list(vt.frustum.shape), image_shape=list(img.shape))
                if layout.get('signature') != signature:
                    raise ValueError('Mapping layout grid/depth/image specification differs')
            else:
                layout = make_layout(metas, vt.grid_size, vt.out_channels, args.mapping_reserve)
                layout['signature'] = dict(grid_lower_bound=vt.grid_lower_bound.cpu().tolist(),
                    grid_interval=vt.grid_interval.cpu().tolist(), frustum_sha256=hashlib.sha256(vt.frustum.cpu().numpy().tobytes()).hexdigest(), frustum_shape=list(vt.frustum.shape),
                    image_shape=list(img.shape))
                output.with_suffix('.mapping.json').write_text(json.dumps(layout, indent=2))
            mapping = make_mapping(metas, layout, device)
            export_inputs = (img,) + mapping
            input_names = ['img'] + mapping_names(layout)
            if args.mapping_only:
                path = output.with_suffix('.inputs.npz')
                np.savez(path, mapping_version=np.array(3, dtype=np.int32), **{name: tensor.detach().cpu().numpy() for name, tensor in zip(input_names, export_inputs)})
                print('Updated mapping/example inputs:', path)
                print('No ONNX was exported; use the existing compiled model and the same layout.')
                return
            wrapper = RuntimeMappingExportModel(model, RuntimeMappingPool(layout)).eval()
        else:
            coor = vt.get_lidar_coor(*prepared[1:7])
            metas = vt.voxel_pooling_prepare_v2(coor)
            if metas[0] is None:
                raise ValueError('Camera geometry projects no valid points into BEV')
            pool = FixedPool(metas, vt.grid_size, vt.out_channels)
            print('Optimized fixed pool:', len(pool.groups), 'buckets;', pool.valid_points,
                  'valid points;', pool.padded_points, 'padded slots')
            wrapper = ExportModel(model, pool).eval()
            export_inputs, input_names = (img,), ['img']
        if args.qat:
            if args.mapping_only:
                p.error('--qat does not apply to --mapping-only')
            # Rebuild exactly the training graph and buffer shapes. Calibration
            # only materializes state; strict loading then restores trained scales.
            model = convert_convolutions(model, cfg.qat, lambda: wrapper(*export_inputs))
            model.load_state_dict(state, strict=True)
            wrapper.to(device).eval()
            from oritek_qat import quantize_tool
            quantize_tool.disable_observer(model)
        outputs = wrapper(*export_inputs)
        if not all(torch.isfinite(t).all() for t in outputs):
            raise ValueError('Nonfinite model output')
        names = [f'det_task{i}_{k}' for i in range(len(model.pts_bbox_head.task_heads)) for k in KEYS] + ['occ_logits']
        # PyTorch 1.13 warns about optional prim::Constant values while tracing.
        # Suppress only this known diagnostic, then validate the serialized ONNX.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=r'The shape inference of prim::Constant type is missing,.*',
                category=UserWarning,
                module=r'torch\.onnx(\..*)?',
            )
            export = quantize_tool.export_onnx if args.qat else torch.onnx.export
            export(wrapper, export_inputs, str(output), opset_version=16,
                   input_names=input_names, output_names=names)
    graph = onnx.load(str(output))
    graph = onnx.shape_inference.infer_shapes(graph, strict_mode=True)
    onnx.checker.check_model(graph, full_check=True)
    onnx.helper.set_model_props(graph, {
        'quantization': 'oritek_qat_int8_conv_islands' if args.qat else 'floating_point',
        'geometry_mode': 'mapping' if args.runtime_mapping else 'fixed',
        'mapping_version': '3' if args.runtime_mapping else 'none',
        'fixed_pool_mode': 'none' if args.runtime_mapping else 'direct',
        'geometry_frame': 'sensor2keyegos maps each camera sensor to the reference key ego frame',
        'occ_layout': 'NCHW; channel=z*num_classes+class'})
    graph.graph.output[-1].doc_string = (
        'NCHW OCC logits [B,Dz*num_classes,Y,X]; channel=z*num_classes+class. '
        'Legacy [B,X,Y,Z,C] = raw.reshape(B,Z,C,Y,X).transpose(0,4,3,1,2).')
    onnx.save(graph, str(output))
    if any(n.domain for n in graph.graph.node):
        raise ValueError('Unexpected custom ONNX operator')
    print('Exported:', output)
    print('OCC output: NCHW [B,Dz*num_classes,Y,X], channel=z*num_classes+class')
    if args.runtime_mapping:
        example_path = output.with_suffix('.inputs.npz')
        np.savez(example_path, mapping_version=np.array(3, dtype=np.int32), **{name: tensor.detach().cpu().numpy()
                                 for name, tensor in zip(input_names, export_inputs)})
        print('Cached mapping inputs:', input_names)
        print('Example inputs:', example_path)
        print('Mapping inputs: FP32 base-1024 indices, zero-depth padding, no masks; preserve mapping precision in SDK quant_tensors.')
        print('Layout:', args.mapping_layout or output.with_suffix('.mapping.json'))
        print('Runtime Gather performance and quantized accuracy must be checked on the SDK.')
    else:
        print('Fixed geometry:', args.geometry_npz or f'validation sample {args.sample_index}')
    for name, tensor in zip(names, outputs):
        print(name, tuple(tensor.shape), tensor.dtype)

if __name__ == '__main__':
    main()
