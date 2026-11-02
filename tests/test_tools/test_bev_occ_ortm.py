"""Protect SDK tensor-name mapping and strict binary decoding."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from bev_occ_inference import add_inference_arguments, parse_inference_arguments
from bev_occ_ortm import read_outputs


def test_ortm_backend_and_quantization_switch():
    parser = argparse.ArgumentParser()
    add_inference_arguments(parser)
    args = parse_inference_arguments(parser, ['config.py', 'g6_fake_quant.ortm'])
    assert args.backend == 'ortm'
    assert args.ortm_provider == 'cuda'
    with pytest.raises(SystemExit):
        parse_inference_arguments(parser, ['config.py', 'g6_fake_quant.ortm', '--disable-fake-quant'])


def test_outputs_use_tensor_names_not_file_order(tmp_path):
    expected = {'det_task0_reg': (1, 2, 2, 3), 'occ_logits': (1, 36, 2, 3)}
    for index, (name, shape) in enumerate(reversed(list(expected.items()))):
        np.full(shape, index + 1, dtype='<f4').tofile(tmp_path / f'{index}_{name}.bin')
    values = read_outputs(tmp_path, expected)
    assert values['occ_logits'].shape == (1, 36, 2, 3)
    assert np.all(values['occ_logits'] == 1)
    assert np.all(values['det_task0_reg'] == 2)


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'fp16', 'nonfinite', 'extra'])
def test_invalid_outputs_are_rejected(tmp_path, problem):
    path = tmp_path / '0_occ_logits.bin'
    value = np.ones((1, 18, 2, 3), dtype='<f4')
    if problem == 'nonfinite':
        value.flat[0] = np.nan
    if problem == 'fp16':
        value = value.astype('<f2')
    if problem != 'missing':
        value.tofile(path)
    if problem == 'duplicate':
        value.tofile(tmp_path / '1_occ_logits.bin')
    if problem == 'extra':
        value.tofile(tmp_path / '1_unknown.bin')
    with pytest.raises(ValueError):
        read_outputs(tmp_path, {'occ_logits': (1, 18, 2, 3)})
