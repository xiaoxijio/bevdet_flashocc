# Single RTX 3090: inspect the original single-frame BEVDet training flow.
_base_ = ["./bevdet-r50.py"]

# Cached ImageNet weights: no network download during model initialization.
model = dict(
    img_backbone=dict(
        pretrained="/home/fjm/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth",
        with_cp=False),
    img_view_transformer=dict(with_cp=False, accelerate=False))

# One sample contains six cameras. Keep loading in the debugger process.
data = dict(samples_per_gpu=1, workers_per_gpu=0)

# Small learning run; lr scaled from the inherited batch-8 setting.
optimizer = dict(lr=2.5e-5)
lr_config = dict(
    _delete_=True, policy="fixed", by_epoch=False,
    warmup="linear", warmup_iters=5, warmup_ratio=0.1)
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=20)
checkpoint_config = dict(by_epoch=False, interval=20, max_keep_ckpts=1)
evaluation = dict(by_epoch=False, interval=20)
log_config = dict(
    _delete_=True, interval=1, hooks=[dict(type="TextLoggerHook")])
custom_hooks = []  # Disable EMA while tracing forward/backward.
fp16 = None       # FP32 makes tensors and gradients easier to inspect.
cudnn_benchmark = False
workflow = [("train", 1)]
work_dir = "./work_dirs/bevdet-r50-3090-debug"
