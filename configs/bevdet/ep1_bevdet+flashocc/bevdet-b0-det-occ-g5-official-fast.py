# Four-GPU measured throughput recipe. Inference graph/weights are unchanged.
# Training GT construction is precomputed in DataLoader workers.
_base_ = ['./bevdet-b0-det-occ-g5-worker-targets-base.py']
data = dict(samples_per_gpu=16, workers_per_gpu=8)
# Smaller initial scale passed the four-GPU probe without the default-scale overflow.
fp16 = dict(_delete_=True,
            loss_scale=dict(init_scale=512., growth_interval=2000))
# Keep the conservative fine-tuning LR; do not pass --autoscale-lr.
optimizer = dict(lr=5e-5)
# Approximately 4000 warmup samples, as in the original batch-eight recipe.
lr_config = dict(warmup_iters=63, step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)
log_config = dict(interval=20)
work_dir = './work_dirs/ep1_g5_official_fast'
