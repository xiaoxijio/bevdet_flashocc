"""One-real-batch QAT smoke check; does not start the full training schedule."""
import argparse
import copy
import os
import sys
sys.path.insert(0, os.getcwd())
import torch
from mmcv import Config
from mmcv.parallel import scatter
from mmcv.runner import build_optimizer
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.utils.oritek_qat import (prepare_training, freeze_bn,
                                     OritekQATHook, qat_tool)
import logging


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/bevdet/ep2_bevdet+flashocc/bevdet-r50-det-occ-bev128-depth-qat.py')
    p.add_argument('--checkpoint-output', default='/tmp/bevdet_qat_smoke.pth')
    a = p.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(2)
    cfg = Config.fromfile(a.config)
    cfg.gpu_ids = [0]
    cfg.seed = 0
    logger = logging.getLogger('mmdet')
    logging.basicConfig(level=logging.INFO)
    # Use the real training pipeline without allocating a CBGS resampling list.
    ds_cfg = copy.deepcopy(cfg.data.train)
    while 'dataset' in ds_cfg:
        ds_cfg = ds_cfg['dataset']
    dataset = build_dataset(ds_cfg)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                              dist=False, shuffle=False)
    batch = scatter(next(iter(loader)), [0])[0]
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    source = torch.load(cfg.load_from, map_location='cpu')
    state = source.get('state_dict', source)
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    # Check Linear -> Conv equivalence with real checkpoint weights and a
    # rectangular feature map to catch accidental X/Y layout swaps.
    fp_cfg = copy.deepcopy(cfg.model)
    fp_cfg.occ_head.use_conv_predicter = False
    old = build_model(fp_cfg, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    old.load_state_dict(state, strict=True)
    old.occ_head.eval(); model.occ_head.eval()
    x = torch.randn(1, cfg.model.occ_head.in_dim, 7, 9)
    with torch.no_grad():
        expected = old.occ_head(x)
        actual = model.occ_head(x)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    print('LINEAR_CONV_EQUIVALENCE_OK',float((actual-expected).abs().max()),flush=True)
    del old, source, state
    model.cuda().eval()
    with torch.no_grad():
        losses = model(return_loss=True, **batch)
        baseline, baseline_logs = model._parse_losses(losses)
        assert torch.isfinite(baseline)
    print('REAL_FP_LOSSES_OK', baseline_logs, flush=True)
    model = prepare_training(model, loader, cfg, logger)
    model.train(); freeze_bn(model)
    optimizer = build_optimizer(model, cfg.optimizer)
    optimizer.zero_grad()
    losses = model(return_loss=True, **batch)
    loss, logs = model._parse_losses(losses)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g).item() for g in gradients)
    optimizer.step()
    print('REAL_QAT_FORWARD_BACKWARD_STEP_OK',logs,'gradients',len(gradients),flush=True)
    model.eval()
    with torch.no_grad():
        before = model(return_loss=True, **batch)
        before = {k:v.detach().clone() if torch.is_tensor(v) else v for k,v in before.items()}
    scales = {n:v.detach().clone() for n,v in model.state_dict().items() if n.endswith(('.scale','.zero_point'))}
    assert scales
    for name, module in model.named_modules():
        if hasattr(module,'observer_enabled'):
            assert not bool(module.observer_enabled[0]), name
    checkpoint = dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                      meta=dict(oritek_qat=dict(format_version=1,recipe=dict(cfg.qat))))
    torch.save(checkpoint,a.checkpoint_output)
    # Overwrite every persisted scale to ensure strict loading restores it.
    for name,value in model.named_buffers():
        if name.endswith('.scale'):
            value.fill_(123)
    model.load_state_dict(torch.load(a.checkpoint_output,map_location='cpu')['state_dict'],strict=True)
    model.cuda()  # SDK load_state_dict restores some quantizer tensors on CPU.
    for name,value in scales.items():
        torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
    with torch.no_grad():
        after = model(return_loss=True, **batch)
    for key in before:
        if torch.is_tensor(before[key]):
            torch.testing.assert_close(after[key],before[key],rtol=0,atol=0)
    print('QAT_STATE_ROUNDTRIP_EVAL_FREEZE_OK',len(scales),flush=True)
    print('SMOKE_CHECKPOINT',a.checkpoint_output,flush=True)


if __name__ == '__main__':
    main()
