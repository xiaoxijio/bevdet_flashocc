# G1: isolate OCC width reduction; shared encoder and detector stay at 256.
_base_ = ['../ep0_bevdet+flashocc/bevdet-r50-det-occ-shared.py']
model = dict(
    img_backbone=dict(pretrained=None, with_cp=False),
    img_view_transformer=dict(with_cp=False, accelerate=False),
    occ_adapter=dict(out_channels=128),
    occ_head=dict(in_dim=128, out_dim=128))
# MMCV non-strict initialization restores matching tensors and reports/skips
# changed shapes. This is fine-tuning initialization, NOT training resume.
load_from = 'ckpt/epoch_20_ema.pth'
resume_from = None
# Four GPUs x two samples = global batch eight; retain the fine-tuning LR.
# Do not pass --autoscale-lr.
data = dict(samples_per_gpu=2, workers_per_gpu=4)
optimizer = dict(lr=5e-5)
lr_config = dict(warmup_iters=500)
custom_hooks = [dict(type='MEGVIIEMAHook', init_updates=0, priority='NORMAL')]
find_unused_parameters = False
fp16 = None
evaluation = dict(interval=2, metric=['bbox', 'occ'])
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'),
                                    dict(type='TensorboardLoggerHook')])
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'
work_dir = './work_dirs/ep1_g1_occ128'
