"""Shared PTH/ONNX inference and postprocessing for evaluation and rendering."""
import copy
import json
from pathlib import Path

import numpy as np

KEYS = ('reg', 'height', 'dim', 'rot', 'vel', 'heatmap')


def add_inference_arguments(parser):
    parser.add_argument('config', help='Matching joint Det/OCC config')
    parser.add_argument('model', help='PTH checkpoint or exported ONNX model')
    parser.add_argument('--backend', choices=('pth', 'onnx'), default='pth')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda',
                        help='PTH and detection postprocessing device')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--onnx-provider', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--mapping-layout', help='Runtime-mapping ONNX layout JSON; auto-detected alongside ONNX')


def load_context(args):
    import torch
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmdet3d.datasets import build_dataset
    if args.workers < 0 or args.threads < 1 or args.gpu_id < 0:
        raise ValueError('workers/gpu-id must be nonnegative; threads must be positive')
    if not Path(args.model).is_file():
        raise FileNotFoundError(args.model)
    if args.backend == 'onnx' and Path(args.model).suffix.lower() != '.onnx':
        raise ValueError('--backend onnx requires an ONNX file')
    if args.backend == 'pth' and args.mapping_layout:
        raise ValueError('--mapping-layout is only used with --backend onnx')
    torch.set_num_threads(args.threads)
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use --device cpu')
        torch.cuda.set_device(args.gpu_id)
    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    cfg.data.test.test_mode = True
    cfg.model.train_cfg = None
    cfg.model.img_backbone.pretrained = None
    return cfg, build_dataset(cfg.data.test)


def make_loader(dataset, indices, workers):
    import torch
    from mmdet3d.datasets import build_dataloader
    return build_dataloader(torch.utils.data.Subset(dataset, indices), samples_per_gpu=1,
                            workers_per_gpu=workers, dist=False, shuffle=False)


def unpack_batch(batch, args):
    from mmcv.parallel import scatter
    return scatter(batch, [args.gpu_id if args.device == 'cuda' else -1])[0]


def occ_from_nchw(raw, dz, num_classes=18):
    """Export channel=z*C+c, NCHW -> class IDs [X,Y,Z], including rectangular grids."""
    raw = np.asarray(raw)
    if raw.ndim != 4 or raw.shape[0] != 1 or raw.shape[1] != dz * num_classes:
        raise ValueError('Expected OCC [1,Dz*num_classes,Y,X], got ' + str(raw.shape))
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite ONNX occupancy logits')
    _, _, y, x = raw.shape
    return raw.reshape(dz, num_classes, y, x).argmax(axis=1).transpose(2, 1, 0).astype(np.uint8)


def checked_occupancy(result, gt):
    prediction = np.asarray(result['occ'])
    if prediction.shape != gt.shape:
        raise ValueError(f'Occupancy shape mismatch: prediction={prediction.shape}, GT={gt.shape}; check config/occ_crop')
    if not np.issubdtype(prediction.dtype, np.integer) or np.any((prediction < 0) | (prediction >= 18)):
        raise ValueError('Expected occupancy class IDs 0..17')
    return prediction


class PthPredictor:
    def __init__(self, args, cfg, dataset, example):
        import torch
        from mmdet3d.models import build_model
        self.args = args
        self.model = build_model(cfg.model, test_cfg=cfg.get('test_cfg')).eval()
        checkpoint = torch.load(args.model, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint)
        state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
        self.device = torch.device(f'cuda:{args.gpu_id}' if args.device == 'cuda' else 'cpu')
        self.model.to(self.device)
        if checkpoint.get('meta', {}).get('oritek_qat'):
            from mmdet3d.utils.oritek_qat import convert_convolutions, validate_recipe, qat_tool
            validate_recipe(checkpoint, cfg.get('qat', {}))
            # Use the export wrapper to initialize quantizer shapes without CUDA-only BEV pooling.
            wrapper = self.cpu_compatible_wrapper(example)
            self.model = convert_convolutions(self.model, cfg.qat, lambda: wrapper(example['img_inputs'][0][0].squeeze(0)))
            self.model.load_state_dict(state, strict=True)
            self.model.to(self.device).eval()
            qat_tool().disable_observer(self.model)
        else:
            self.model.load_state_dict(state, strict=True)
        self.model.CLASSES = dataset.CLASSES
        self.model.eval()
        self.description = 'PTH native inference' if args.device == 'cuda' else 'PTH CPU: same weights, PyTorch fixed-pool decomposition per frame'

    def cpu_compatible_wrapper(self, data):
        from tools.export_bev_occ_onnx import ExportModel, FixedPool
        prepared = self.model.prepare_inputs(data['img_inputs'][0])
        vt = self.model.img_view_transformer
        if int(vt.grid_size[2]) != 1:
            raise ValueError('CPU PTH fallback supports a single BEV Z bin')
        metas = vt.voxel_pooling_prepare_v2(vt.get_lidar_coor(*prepared[1:7]))
        if metas[0] is None:
            raise ValueError('No points project into BEV')
        return ExportModel(self.model, FixedPool(metas, vt.grid_size, vt.out_channels)).eval()

    def __call__(self, data):
        import torch
        if self.args.device == 'cuda':
            with torch.no_grad():
                return self.model(return_loss=False, rescale=True, **data)[0]
        wrapper = self.cpu_compatible_wrapper(data)
        with torch.no_grad():
            outputs = wrapper(data['img_inputs'][0][0].squeeze(0))
        names = [f'det_task{i}_{k}' for i in range(len(self.model.pts_bbox_head.task_heads)) for k in KEYS]
        values = dict(zip(names + ['occ_logits'], outputs))
        return decode_outputs(values, self.model.pts_bbox_head, data['img_metas'][0],
                              self.model.occ_head.Dz, self.model.occ_head.num_classes, self.device)


def decode_outputs(values, head, img_metas, dz, num_classes, device):
    import torch
    from mmdet3d.core import bbox3d2result
    tasks = []
    for i in range(len(head.task_heads)):
        task = {}
        for key in KEYS:
            name = f'det_task{i}_{key}'
            if name not in values:
                raise ValueError('Missing ONNX detection output: ' + name)
            value = values[name]
            tensor = value.detach().to(device=device, dtype=torch.float32) if torch.is_tensor(value) else torch.from_numpy(np.asarray(value)).to(device=device, dtype=torch.float32)
            if tensor.ndim != 4 or tensor.shape[0] != 1 or not bool(torch.isfinite(tensor).all()):
                raise ValueError('Invalid detection output: ' + name)
            task[key] = tensor
        tasks.append([task])
    with torch.no_grad():
        boxes, scores, labels = head.get_bboxes(tasks, img_metas, rescale=True)[0]
    logits = values['occ_logits']
    if torch.is_tensor(logits):
        logits = logits.detach().cpu().numpy()
    return dict(pts_bbox=bbox3d2result(boxes, scores, labels),
                occ=occ_from_nchw(logits, dz, num_classes))


class OnnxPredictor:
    def __init__(self, args, cfg, dataset, example):
        import torch
        import onnxruntime as ort
        from mmdet3d.models import build_head
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu_id}' if args.device == 'cuda' else 'cpu')
        head_cfg = copy.deepcopy(cfg.model.pts_bbox_head)
        test_cfg = cfg.model.get('test_cfg', cfg.get('test_cfg', {}))
        head_cfg['test_cfg'] = test_cfg.get('pts', test_cfg)
        head_cfg['train_cfg'] = None
        self.head = build_head(head_cfg).to(self.device).eval()
        self.dz = cfg.model.occ_head.Dz
        self.num_classes = cfg.model.occ_head.num_classes
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        providers = ['CPUExecutionProvider']
        if args.onnx_provider == 'cuda':
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('CUDA ONNX Runtime provider unavailable; use --onnx-provider cpu or install matching onnxruntime-gpu')
            providers.insert(0, ('CUDAExecutionProvider', {'device_id': args.gpu_id}))
        self.session = ort.InferenceSession(args.model, sess_options=options, providers=providers)
        self.inputs = self.session.get_inputs()
        self.output_names = [o.name for o in self.session.get_outputs()]
        expected = {f'det_task{i}_{k}' for i in range(len(self.head.task_heads)) for k in KEYS} | {'occ_logits'}
        if set(self.output_names) != expected:
            raise ValueError('ONNX outputs do not match config: expected ' + str(sorted(expected)))
        if not self.inputs or self.inputs[0].name != 'img':
            raise ValueError('Expected this project exporter ONNX with first input named img')
        self.layout = self.vt = None
        if len(self.inputs) > 1:
            from mmdet3d.models import build_neck
            from tools.bev_occ_mapping import mapping_names
            path = Path(args.mapping_layout) if args.mapping_layout else Path(args.model).with_suffix('.mapping.json')
            self.layout = json.loads(path.read_text())
            if [i.name for i in self.inputs] != ['img'] + mapping_names(self.layout):
                raise ValueError('Mapping layout does not match ONNX input names/order')
            self.vt = build_neck(cfg.model.img_view_transformer).cpu().eval()
        elif args.mapping_layout:
            raise ValueError('Single-input fixed-geometry ONNX does not consume a mapping layout')
        self.description = ('ONNX Runtime; dynamic geometry from per-frame runtime maps' if self.layout else
                            'ONNX Runtime; fixed geometry baked into ONNX (not recomputed for each sample)')
        print('ONNX providers:', self.session.get_providers())

    def __call__(self, data):
        import torch
        images = data['img_inputs'][0][0]
        if images.shape[0] != 1:
            raise ValueError('Only batch size 1 is supported')
        feed = {'img': np.ascontiguousarray(images[0].detach().cpu().numpy(), dtype=np.float32)}
        if self.layout:
            from tools.bev_occ_mapping import make_mapping, mapping_names
            from mmdet3d.models.detectors.bevdet import BEVDet
            inputs = [x.detach().cpu() for x in data['img_inputs'][0]]
            prepared = BEVDet.prepare_inputs(None, inputs)
            with torch.no_grad():
                metas = self.vt.voxel_pooling_prepare_v2(self.vt.get_lidar_coor(*prepared[1:7]))
            maps = make_mapping(metas, self.layout)
            feed.update({name: np.ascontiguousarray(x.numpy()) for name, x in zip(mapping_names(self.layout), maps)})
        for descriptor in self.inputs:
            value = feed[descriptor.name]
            if descriptor.type not in ('tensor(float)', 'tensor(float16)'):
                raise ValueError('Unsupported ONNX input type: ' + descriptor.type)
            feed[descriptor.name] = value.astype(np.float16 if descriptor.type == 'tensor(float16)' else np.float32, copy=False)
            if len(value.shape) != len(descriptor.shape) or any(isinstance(w, int) and a != w for a, w in zip(value.shape, descriptor.shape)):
                raise ValueError(f'ONNX input {descriptor.name} shape mismatch: {value.shape} vs {descriptor.shape}')
        values = dict(zip(self.output_names, self.session.run(self.output_names, feed)))
        return decode_outputs(values, self.head, data['img_metas'][0], self.dz, self.num_classes, self.device)


def make_predictor(args, cfg, dataset, example):
    return (PthPredictor if args.backend == 'pth' else OnnxPredictor)(args, cfg, dataset, example)
