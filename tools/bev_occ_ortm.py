"""SDK fake-quant ORTM forward pass; shares BEVDet preprocessing and decoding."""
import copy
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def read_outputs(directory, descriptors):
    """Read runner FP32 binaries by tensor name, never by directory order."""
    values = {}
    files = list(Path(directory).glob('*.bin'))
    for name, shape in descriptors.items():
        matches = [p for p in files if p.name == name + '.bin' or
                   (p.stem.partition('_')[0].isdigit() and p.stem.partition('_')[2] == name)]
        if len(matches) != 1:
            raise ValueError(f'Expected one ORTM output for {name}; found {matches}')
        path = matches[0]
        expected = int(np.prod(shape)) * 4
        if path.stat().st_size != expected:
            raise ValueError(f'ORTM output {name}: expected FP32 {shape} ({expected} bytes), got {path.stat().st_size} bytes')
        value = np.fromfile(path, dtype='<f4').reshape(shape)
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite ORTM output: ' + name)
        values[name] = value
    if len(files) != len(descriptors):
        raise ValueError('Unexpected ORTM output files: ' + str(files))
    return values


class OrtmPredictor:
    def __init__(self, args, cfg, dataset, example):
        import onnx
        import torch
        from mmdet3d.models import build_head
        from bev_occ_inference import KEYS
        from bev_occ_reporting import file_identity

        self.args = args
        self.model = Path(args.model).resolve()
        if not self.model.stem.endswith('_fake_quant'):
            raise ValueError('oritek_graph_runner requires *_fake_quant.ortm; use the simulator/board runner for a deployment ORTM')
        if not np.isfinite(args.ortm_timeout) or args.ortm_timeout <= 0:
            raise ValueError('--ortm-timeout must be positive and finite')
        repo = Path(__file__).resolve().parents[1]
        root = Path(args.npu_toolchain or os.environ.get('NPU_TOOLCHAIN_PATH') or repo.parent / 'npu_toolchain').resolve()
        self.runner = root / 'linux/simulator/bin/oritek_graph_runner'
        if not self.runner.is_file() or not os.access(self.runner, os.X_OK):
            raise FileNotFoundError(f'SDK runner not executable: {self.runner}; set --npu-toolchain')
        basename = self.model.stem[:-len('_fake_quant')] + '.onnx'
        if args.ortm_onnx:
            source = Path(args.ortm_onnx).resolve()
        else:
            candidates = [self.model.with_name(basename), repo / 'onnx' / basename]
            source = next((p for p in candidates if p.is_file()), None)
            if source is None:
                raise FileNotFoundError('Matching source ONNX needed for tensor metadata; pass --ortm-onnx')
        graph = onnx.load(str(source), load_external_data=False).graph
        initializer_names = {v.name for v in graph.initializer}
        inputs = [v for v in graph.input if v.name not in initializer_names]
        if len(inputs) != 1 or inputs[0].name != 'img':
            raise ValueError('ORTM backend currently supports only single-input fixed-geometry models')

        def shape_of(value):
            shape = tuple(d.dim_value for d in value.type.tensor_type.shape.dim)
            if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT or not shape or any(d <= 0 for d in shape):
                raise ValueError('ORTM metadata requires static FP32 tensor: ' + value.name)
            return shape

        self.input_shape = shape_of(inputs[0])
        self.outputs = {v.name: shape_of(v) for v in graph.output}
        self.device = torch.device(f'cuda:{args.gpu_id}' if args.device == 'cuda' else 'cpu')
        head_cfg = copy.deepcopy(cfg.model.pts_bbox_head)
        test_cfg = cfg.model.get('test_cfg', cfg.get('test_cfg', {}))
        head_cfg['test_cfg'] = test_cfg.get('pts', test_cfg)
        head_cfg['train_cfg'] = None
        self.head = build_head(head_cfg).to(self.device).eval()
        expected = {f'det_task{i}_{k}' for i in range(len(self.head.task_heads)) for k in KEYS} | {'occ_logits'}
        if set(self.outputs) != expected:
            raise ValueError('ORTM source ONNX outputs do not match detection config: ' + str(sorted(expected)))
        self.dz = cfg.model.occ_head.Dz
        self.num_classes = cfg.model.occ_head.num_classes
        self.env = os.environ.copy()
        libraries = [str(root / 'linux/simulator/lib')]
        # Same CUDA/cuDNN combination used by Compilation/app/bev_occ/env.sh.
        cuda_lib = Path('/usr/local/cuda-11.8/lib64')
        if (cuda_lib / 'libcudnn.so').is_file():
            libraries.insert(0, str(cuda_lib))
            self.env['CUDA_HOME'] = str(cuda_lib.parent)
            self.env['PATH'] = str(cuda_lib.parent / 'bin') + ':' + self.env.get('PATH', '')
        if self.env.get('LD_LIBRARY_PATH'):
            libraries.append(self.env['LD_LIBRARY_PATH'])
        self.env['LD_LIBRARY_PATH'] = ':'.join(libraries)
        self.description = ('SDK fake-quant ORTM quantization-loss evaluation; fixed geometry; '
                            f'{args.ortm_provider}; source ONNX used only for tensor metadata; '
                            'one SDK process per frame')
        self.metadata = dict(ortm_runner=file_identity(self.runner), ortm_source_onnx=file_identity(source),
                             ortm_input_shape=self.input_shape, ortm_output_shapes=self.outputs,
                             npu_toolchain=str(root))

    def __call__(self, data):
        from bev_occ_inference import decode_outputs
        images = data['img_inputs'][0][0]
        if images.shape[0] != 1:
            raise ValueError('Only batch size 1 is supported')
        image = np.ascontiguousarray(images[0].detach().cpu().numpy(), dtype='<f4')
        if image.shape != self.input_shape or not np.isfinite(image).all():
            raise ValueError(f'Invalid ORTM input {image.shape}; expected {self.input_shape} with finite values')
        with tempfile.TemporaryDirectory(prefix='bev-occ-ortm-') as temporary:
            work = Path(temporary)
            input_path, output_path = work / 'img.bin', work / 'outputs'
            image.tofile(input_path)
            command = [str(self.runner), '--model', str(self.model), '--input', str(input_path),
                       '--output', str(output_path), '--gpu_id',
                       str(self.args.gpu_id if self.args.ortm_provider == 'cuda' else -1),
                       '--thread_num', str(self.args.threads), '--log_level', 'error']
            with (work / 'runner.log').open('w+') as log:
                try:
                    result = subprocess.run(command, cwd=work, env=self.env, stdout=log,
                                            stderr=subprocess.STDOUT, timeout=self.args.ortm_timeout)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(f'ORTM SDK inference timed out after {self.args.ortm_timeout}s') from exc
                log.seek(0, 2)
                size = log.tell()
                log.seek(max(0, size - 6000))
                details = log.read()
            if result.returncode:
                raise RuntimeError(f'ORTM SDK runner exited {result.returncode}:\n{details}')
            try:
                values = read_outputs(output_path, self.outputs)
            except ValueError as exc:
                raise RuntimeError(f'{exc}\nSDK log:\n{details}') from exc
        return decode_outputs(values, self.head, data['img_metas'][0], self.dz, self.num_classes, self.device)
