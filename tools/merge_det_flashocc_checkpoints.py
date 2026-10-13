"""Build validated initialization weights for shared-BEV Det+FlashOCC.

This is an initialization checkpoint, not an optimizer/runner resume file.
Existing output files are never overwritten.
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mmcv import Config
from mmdet3d.models import build_model

DET_PREFIXES = ('img_backbone.', 'img_neck.', 'img_view_transformer.',
                'img_bev_encoder_backbone.', 'img_bev_encoder_neck.', 'pts_bbox_head.')


def state(path):
    raw = torch.load(path, map_location='cpu')
    raw = raw.get('state_dict', raw)
    return {key[7:] if key.startswith('module.') else key: value
            for key, value in raw.items()}


def sha256(path):
    checksum = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--det', required=True)
    parser.add_argument('--occ', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(str(output))
    model = build_model(Config.fromfile(args.config).model)
    target = model.state_dict()
    sources = {'det': state(args.det), 'occ': state(args.occ)}
    merged = {}
    counts = {'det': 0, 'occ': 0}
    for key, tensor in target.items():
        if key.startswith('occ_adapter.'):
            continue
        if key.startswith(DET_PREFIXES):
            source = 'det'
        elif key.startswith('occ_head.'):
            source = 'occ'
        else:
            raise ValueError('No explicit initialization policy for ' + key)
        if key not in sources[source]:
            raise KeyError(source + ' checkpoint missing ' + key)
        value = sources[source][key]
        if value.shape != tensor.shape:
            raise ValueError('{}: source {} != target {}'.format(
                key, tuple(value.shape), tuple(tensor.shape)))
        merged[key] = value
        counts[source] += 1
    incompatible = model.load_state_dict(merged, strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.startswith('occ_adapter.') for key in incompatible.missing_keys)
    # Serialize only pretrained tensors: adapter initializes via the model as usual.
    checkpoint = dict(state_dict=merged, meta=dict(
        initialization_only=True, config=args.config,
        det_source=args.det, det_sha256=sha256(args.det),
        occ_source=args.occ, occ_sha256=sha256(args.occ),
        parameter_counts=counts, newly_initialized=incompatible.missing_keys))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(output))
    print('Saved:', output)
    print('Loaded tensors:', counts)
    print('Freshly initialized:', incompatible.missing_keys)


if __name__ == '__main__':
    main()
