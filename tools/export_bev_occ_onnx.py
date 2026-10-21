"""Export joint Det/OCC with optional runtime geometry and NCHW OCC logits."""
import argparse
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

KEYS = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']

class FixedPool(torch.nn.Module):
    """Sum each voxel directly; avoid OMG's dense CumSum-to-MatMul expansion."""
    def __init__(self, metas, grid_size, channels):
        super().__init__()
        rb, rd, rf, starts, lengths = metas
        self.height, self.width = int(grid_size[1]), int(grid_size[0])
        self.channels = channels
        self.register_buffer('rd', rd.long())
        self.register_buffer('rf', rf.long())
        self.groups = []
        ordered_bev = []
        max_length = int(lengths.max().item())
        bound = 1
        while bound < 2 * max_length:
            ids = torch.where((lengths <= bound) & (lengths > (bound // 2 if bound > 1 else 0)))[0]
            if ids.numel():
                offsets = torch.arange(bound, device=starts.device)[None, :]
                valid = offsets < lengths[ids, None]
                indices = torch.where(valid, starts[ids, None] + offsets, 0).long()
                self.register_buffer(f'indices_{bound}', indices.reshape(-1))
                self.register_buffer(f'mask_{bound}', valid[:, :, None].float())
                self.groups.append((bound, int(ids.numel())))
                ordered_bev.append(rb[starts[ids].long()])
            bound *= 2
        self.register_buffer('bev_indices', torch.cat(ordered_bev).long())

    def forward(self, depth, feat):
        values = depth.reshape(-1, 1)[self.rd] * feat.reshape(-1, self.channels)[self.rf]
        groups = []
        for bound, count in self.groups:
            x = values[getattr(self, f'indices_{bound}')].reshape(count, bound, self.channels)
            x = x * getattr(self, f'mask_{bound}')
            # Explicit reshape is required by OMG's ReduceSum shape inference.
            groups.append(x.sum(dim=1).reshape(count, self.channels))
        sums = torch.cat(groups, dim=0)
        bev = feat.new_zeros((self.height * self.width, self.channels))
        bev[self.bev_indices] = sums
        return bev.reshape(1, self.height, self.width, self.channels).permute(0, 3, 1, 2).contiguous()

class _ScatterSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, indices, values):
        return base.index_add(0, indices.reshape(-1), values)

    @staticmethod
    def symbolic(graph, base, indices, values):
        # Duplicate voxel indices MUST accumulate, not overwrite. Opset >=16.
        return graph.op('ScatterND', base, indices, values,
                        reduction_s='add').setType(base.type())


class RuntimeGeometryPool(torch.nn.Module):
    """Batch=1 LSS with runtime geometry and standard ONNX ScatterND-add.

    The number/order of cameras, image size, depth bins and BEV grid stay fixed.
    Calibration, augmentation and projected voxel membership remain runtime data.
    No data-dependent NonZero/Sort or frozen projection indices are used.
    """
    def __init__(self, frustum, grid_lower_bound, grid_interval, grid_size, channels):
        super().__init__()
        self.height, self.width = int(grid_size[1]), int(grid_size[0])
        if int(grid_size[2]) != 1:
            raise ValueError('Runtime geometry supports a single BEV Z bin')
        self.channels = channels
        self.register_buffer('frustum', frustum.detach().clone())
        self.register_buffer('lower_bound', grid_lower_bound.detach().clone())
        self.register_buffer('interval', grid_interval.detach().clone())
        self.register_buffer('grid_size', grid_size.detach().clone().long())
        self.register_buffer('empty_bev', torch.zeros(self.height * self.width, channels))

    @staticmethod
    def apply_matrix3(matrix, points):
        # SDK dynamic batch.gemm cannot compile the skinny 3x3 ray transforms.
        # Explicit scalar components keep broadcasts at rank 2 and use Mul/Add.
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        rows = []
        for row in range(3):
            rows.append(matrix[:, row, 0:1] * x
                        + matrix[:, row, 1:2] * y
                        + matrix[:, row, 2:3] * z)
        return torch.stack(rows, dim=-1)

    def coordinates(self, sensor2keyegos, intrins_inv, post_rots_inv, post_trans, bda):
        cameras = sensor2keyegos.shape[1]
        points = self.frustum.reshape(1, -1, 3).expand(cameras, -1, -1)
        points = points - post_trans.reshape(cameras, 1, 3)
        points = self.apply_matrix3(post_rots_inv.reshape(cameras, 3, 3), points)
        points = torch.cat((points[..., :2] * points[..., 2:3],
                            points[..., 2:3]), dim=-1)
        rotation = sensor2keyegos[:, :, :3, :3].reshape(cameras, 3, 3)
        intrinsic = intrins_inv.reshape(cameras, 3, 3)
        rows = []
        for row in range(3):
            rows.append(rotation[:, row, 0:1] * intrinsic[:, 0, :]
                        + rotation[:, row, 1:2] * intrinsic[:, 1, :]
                        + rotation[:, row, 2:3] * intrinsic[:, 2, :])
        combine = torch.stack(rows, dim=1)
        points = self.apply_matrix3(combine, points)
        points = points + sensor2keyegos[:, :, :3, 3].reshape(cameras, 1, 3)
        points = self.apply_matrix3(bda[:, :3, :3], points.reshape(1, -1, 3))
        return points.reshape(-1, 3) + bda[0, :3, 3]

    def forward(self, depth, feat, sensor2keyegos, intrins_inv, post_rots_inv, post_trans, bda):
        points = self.coordinates(sensor2keyegos, intrins_inv, post_rots_inv, post_trans, bda)
        # Match the training implementation's cast (truncate towards zero).
        voxels = ((points - self.lower_bound) / self.interval).long().reshape(-1, 3)
        # SDK 2.1.6 cannot lower GreaterOrEqual/And/Not. Integer i>=0
        # equals i>-1; multiply exact 0/1 masks instead of logical reductions.
        vx, vy, vz = voxels[:, 0], voxels[:, 1], voxels[:, 2]
        valid = ((vx > -1).to(feat.dtype) * (vx < self.width).to(feat.dtype)
                 * (vy > -1).to(feat.dtype) * (vy < self.height).to(feat.dtype)
                 * (vz > -1).to(feat.dtype) * (vz < 1).to(feat.dtype))
        indices = voxels[:, 1] * self.width + voxels[:, 0]
        # Invalid rays contribute zero to a safe index, including all-outside rigs.
        indices = torch.where(valid > 0, indices, torch.zeros_like(indices)).reshape(-1, 1)
        # Lift in [N*H*W,D,C], then restore N,D,H,W point ordering for pooling.
        # Avoid the old rank-5 broadcast [N,D,H,W,1] * [N,1,H,W,C].
        cameras, bins = depth.shape[:2]
        pixels = depth.shape[2] * depth.shape[3]
        probability = depth.reshape(cameras, bins, pixels).transpose(1, 2).reshape(-1, bins, 1)
        values = probability * feat.reshape(-1, 1, self.channels)
        values = values.reshape(cameras, pixels, bins, self.channels).permute(0, 2, 1, 3)
        values = values.reshape(-1, self.channels)
        values = values * valid.to(values.dtype).unsqueeze(-1)
        bev = _ScatterSum.apply(self.empty_bev.to(values), indices, values)
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
        if head.use_predicter:
            predictor = head.predicter
            if (not isinstance(predictor, torch.nn.Sequential) or len(predictor) != 3
                    or not isinstance(predictor[0], torch.nn.Linear)
                    or not isinstance(predictor[1], torch.nn.Softplus)
                    or not isinstance(predictor[2], torch.nn.Linear)
                    or predictor[2].out_features != head.Dz * head.num_classes):
                raise ValueError('NCHW export requires Linear/Softplus/Linear OCC predictor')

    def forward(self, features):
        head = self.head
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
        depth, feat = self.image_features(img)
        return self.decode_bev(self.pool(depth, feat))


class RuntimeGeometryExportModel(ExportModel):
    def forward(self, img, sensor2keyegos, intrins_inv, post_rots_inv, post_trans, bda):
        depth, feat = self.image_features(img)
        return self.decode_bev(self.pool(depth, feat, sensor2keyegos, intrins_inv,
                                         post_rots_inv, post_trans, bda))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # TODO 在此修改路径
    p.add_argument('--config', default='configs/bevdet/ep0_bevdet+flashocc/bevdet-r50-det-occ-shared-2gpu-32gb-pretrained.py')
    p.add_argument('--checkpoint', default='ckpt/epoch_20_ema.pth')
    p.add_argument('--output', default='onnx/bev_occ_dynamic.onnx')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Device used for export; does not change ONNX inputs')
    p.add_argument('--dynamic-geometry', action='store_true',
                   help='Use runtime sensor2keyegos/intrins_inv/post_rots_inv/post_trans/bda inputs; requires ScatterND-add support')
    p.add_argument('--sample-index', type=int, default=0, help='Sample for fixed geometry, or example runtime inputs in dynamic mode')
    p.add_argument('--geometry-npz', help='Optional real-rig geometry: sensor2egos/ego2globals/intrins/post_rots/post_trans/bda, batch=1')
    args = p.parse_args()
    if args.sample_index < 0:
        p.error('sample-index must be nonnegative')
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
        if args.dynamic_geometry:
            # sensor2keyegos includes the frame/pose alignment from prepare_inputs.
            # Raw sensor2egos from six asynchronously captured images is NOT equivalent.
            geometry_args = (prepared[1], prepared[3], prepared[4], prepared[5], prepared[6])
            expected_shapes = [(1, img.shape[0], 4, 4), (1, img.shape[0], 3, 3),
                               (1, img.shape[0], 3, 3), (1, img.shape[0], 3), (1, 4, 4)]
            for tensor, shape in zip(geometry_args, expected_shapes):
                if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
                    raise ValueError('Invalid runtime geometry shape/values: ' + str(tuple(tensor.shape)))
            for matrix in (geometry_args[1], geometry_args[2]):
                if (torch.linalg.det(matrix).abs() < 1e-10).any():
                    raise ValueError('Runtime intrins/post_rots must be nonsingular')
            # Invert in preprocessing with the same library as the original
            # model. These inverses remain runtime inputs, never constants.
            geometry_args = (geometry_args[0], torch.inverse(geometry_args[1]),
                             torch.inverse(geometry_args[2]), geometry_args[3], geometry_args[4])
            wrapper = RuntimeGeometryExportModel(model, RuntimeGeometryPool(
                vt.frustum, vt.grid_lower_bound, vt.grid_interval, vt.grid_size,
                vt.out_channels)).to(device).eval()
            export_inputs = (img,) + geometry_args
            input_names = ['img', 'sensor2keyegos', 'intrins_inv', 'post_rots_inv', 'post_trans', 'bda']
        else:
            coor = vt.get_lidar_coor(*prepared[1:7])
            metas = vt.voxel_pooling_prepare_v2(coor)
            if metas[0] is None:
                raise ValueError('Camera geometry projects no valid points into BEV')
            wrapper = ExportModel(model, FixedPool(metas, vt.grid_size, vt.out_channels)).eval()
            export_inputs, input_names = (img,), ['img']
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
            torch.onnx.export(wrapper, export_inputs, str(output), opset_version=16,
                              input_names=input_names, output_names=names)
    graph = onnx.load(str(output))
    graph = onnx.shape_inference.infer_shapes(graph, strict_mode=True)
    onnx.checker.check_model(graph, full_check=True)
    onnx.helper.set_model_props(graph, {
        'geometry_mode': 'runtime' if args.dynamic_geometry else 'fixed',
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
    if args.dynamic_geometry:
        example_path = output.with_suffix('.inputs.npz')
        np.savez(example_path, **{name: tensor.detach().cpu().numpy()
                                 for name, tensor in zip(input_names, export_inputs)})
        print('Runtime geometry inputs:', input_names)
        print('Example inputs:', example_path)
        print('Requires ONNX ScatterND reduction=add; OMG compatibility must be checked.')
    else:
        print('Fixed geometry:', args.geometry_npz or f'validation sample {args.sample_index}')
    for name, tensor in zip(names, outputs):
        print(name, tuple(tensor.shape))

if __name__ == '__main__':
    main()
