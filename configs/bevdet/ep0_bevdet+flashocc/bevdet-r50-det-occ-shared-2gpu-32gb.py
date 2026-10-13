# Single-frame shared-BEV Det+FlashOCC: two GPUs, 32 GiB per GPU.
_base_ = ['./bevdet-r50-det-occ-shared.py']

# Four samples/device -> global batch eight. Start conservatively.
data = dict(samples_per_gpu=4, workers_per_gpu=4)
model = dict(img_backbone=dict(with_cp=False),
             img_view_transformer=dict(with_cp=False, accelerate=False))
find_unused_parameters = False

# Scaled from single-GPU batch 2/lr 5e-5. Do not pass --autoscale-lr.
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(_delete_=True, policy='step', warmup='linear',
                 warmup_iters=500, warmup_ratio=0.001, step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)

fp16 = None
custom_hooks = [dict(type='MEGVIIEMAHook', init_updates=0, priority='NORMAL')]
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
evaluation = dict(interval=2, metric=['bbox', 'occ'])
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'),
                                    dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'
work_dir = './work_dirs/bevdet-r50-det-occ-shared-2gpu-32gb'
load_from = None
resume_from = None
