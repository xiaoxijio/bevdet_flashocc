"""Backend layout and headless volumetric rendering regressions."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pytest
tools_dir = Path(__file__).resolve().parents[2] / 'tools'
sys.path.insert(0, str(tools_dir if tools_dir.is_dir() else Path(__file__).resolve().parent))
import bev_occ_inference as inference
import bev_occ_visualization as vis


def test_nchw_occ_decode_does_not_swap_xy_or_height():
    dz, classes, y, x = 3, 18, 4, 7
    expected = np.arange(x*y*dz).reshape(x,y,dz) % classes
    raw = np.full((1,dz*classes,y,x), -10., dtype=np.float32)
    for i,j,k in np.ndindex(expected.shape):
        raw[0,k*classes+expected[i,j,k],j,i] = 10
    np.testing.assert_array_equal(inference.occ_from_nchw(raw,dz,classes),expected)
    with pytest.raises(ValueError): inference.occ_from_nchw(raw,4,classes)
    raw[0,0,0,0] = np.nan
    with pytest.raises(ValueError): inference.occ_from_nchw(raw,dz,classes)


@pytest.mark.parametrize("model,expected", [
    ("weights.pth", "pth"), ("model.onnx", "onnx"),
    ("dir/MODEL.PTH", "pth"), ("dir/MODEL.ONNX", "onnx"),
])
def test_backend_is_inferred_from_model_suffix(model, expected):
    parser = argparse.ArgumentParser()
    inference.add_inference_arguments(parser)
    args = inference.parse_inference_arguments(parser, ["config.py", model])
    assert args.backend == expected and args.model == model


def test_unknown_model_suffix_and_removed_backend_switch_are_rejected():
    parser = argparse.ArgumentParser()
    inference.add_inference_arguments(parser)
    for argv in (["config.py", "model.bin"], ["config.py", "model.pth.tar"],
                 ["config.py", "model.onnx", "--backend", "pth"]):
        with pytest.raises(SystemExit) as error:
            inference.parse_inference_arguments(parser, argv)
        assert error.value.code == 2


def test_3d_renderer_is_masked_and_shows_height():
    labels=np.full((16,16,4),17,dtype=np.uint8)
    mask=np.ones_like(labels,dtype=bool)
    empty=vis.occupancy_3d_panel('OCC',labels,mask,3.2)
    labels[5:8,6:10,0]=4
    low=vis.occupancy_3d_panel('OCC',labels,mask,3.2)
    labels[5:8,6:10,1:4]=4
    high=vis.occupancy_3d_panel('OCC',labels,mask,3.2)
    assert low.shape==high.shape==(640,640,3)
    assert np.count_nonzero(low!=empty)>100
    assert np.count_nonzero(high!=low)>100
    np.testing.assert_array_equal(vis.occupancy_3d_panel('OCC',labels,np.zeros_like(mask),3.2),empty)


def test_entrypoints_separate_evaluation_and_visualization():
    import subprocess
    root = Path(vis.__file__).parent
    validation = subprocess.check_output([sys.executable, str(root/'validate_bev_occ.py'), '--help'], text=True)
    rendering = subprocess.check_output([sys.executable, str(root/'visualize_bev_occ.py'), '--help'], text=True)
    assert '--backend' not in validation and '--backend' not in rendering
    assert '--vis-only' not in validation and '--scene' not in validation
    assert '--skip-det-eval' not in rendering and '--scene' in rendering


def test_fake_quant_flag_is_explicit_and_rejects_onnx():
    parser = argparse.ArgumentParser()
    inference.add_inference_arguments(parser)
    default = inference.parse_inference_arguments(parser, ["config.py", "qat.pth"])
    disabled = inference.parse_inference_arguments(parser, ["config.py", "qat.pth", "--disable-fake-quant"])
    assert not default.disable_fake_quant
    assert disabled.disable_fake_quant
    with pytest.raises(SystemExit) as error:
        inference.parse_inference_arguments(parser, ["config.py", "model.onnx", "--disable-fake-quant"])
    assert error.value.code == 2
