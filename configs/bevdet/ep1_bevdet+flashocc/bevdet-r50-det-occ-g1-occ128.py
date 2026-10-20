# G1: isolate OCC width reduction; shared encoder and detector stay at 256.
_base_ = ['../ep0_bevdet+flashocc/bevdet-r50-det-occ-shared-2gpu-32gb-pretrained.py']
model = dict(
    occ_adapter=dict(out_channels=128),
    occ_head=dict(in_dim=128, out_dim=128))
# MMCV non-strict initialization restores matching tensors and reports/skips
# changed shapes. This is fine-tuning initialization, NOT training resume.
load_from = 'ckpt/epoch_20_ema.pth'
resume_from = None
optimizer = dict(lr=5e-5)
work_dir = './work_dirs/ep1_g1_occ128'
