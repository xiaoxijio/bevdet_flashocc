# Learning/debug: one sample, CPU data loading, 20 iterations, no EMA.
_base_ = ['./bevdet-r50-det-occ-shared.py']
model = dict(
    img_backbone=dict(with_cp=False,
        pretrained='/home/fjm/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth'),
    img_view_transformer=dict(with_cp=False, accelerate=False))
data = dict(samples_per_gpu=1, workers_per_gpu=0)
# Keep inherited CBGS/pipeline for consistency; only reduce logging/training length.
optimizer = dict(lr=2.5e-5)
lr_config = dict(_delete_=True, policy='fixed', by_epoch=False,
                warmup='linear', warmup_iters=5, warmup_ratio=.1)
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=20)
evaluation = dict(by_epoch=False, interval=20, metric=['bbox','occ'])
checkpoint_config = dict(by_epoch=False, interval=20, max_keep_ckpts=1)
log_config = dict(_delete_=True, interval=1, hooks=[dict(type='TextLoggerHook')])
custom_hooks = []
fp16 = None
work_dir = './work_dirs/bevdet-r50-det-occ-shared-debug'
