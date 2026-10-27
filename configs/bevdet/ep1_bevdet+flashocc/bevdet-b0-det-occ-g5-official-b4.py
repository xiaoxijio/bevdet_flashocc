# Larger per-GPU batch for the same G5 network.
# Fine-tuning LR stays 5e-5; global batch changes, so compare validation metrics.
# Warmup sample count stays 4000 across four GPUs.
_base_ = ['./bevdet-b0-det-occ-g5-official.py']
data = dict(samples_per_gpu=4, workers_per_gpu=4)
lr_config = dict(warmup_iters=250)
work_dir = './work_dirs/ep1_g5_official_b4'
