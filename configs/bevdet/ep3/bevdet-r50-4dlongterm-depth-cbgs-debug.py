# Single RTX 3090 debug: pure BEVDepth4D detection, current + 8 history frames.
_base_ = ['../bevdet-r50-4dlongterm-depth-cbgs.py']

# Keep the original temporal architecture, depth supervision and CBGS.
data = dict(samples_per_gpu=1, workers_per_gpu=2)
model = dict(with_prev=True)
# Replace the inherited SequentialControlHook, which disables history at startup.
custom_hooks = [dict(type='MEGVIIEMAHook', init_updates=0, priority='NORMAL')]
runner = dict(type='EpochBasedRunner', max_epochs=2)
lr_config = dict(warmup_iters=10, step=[1])
log_config = dict(interval=1)
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
evaluation = dict(interval=2)
load_from = None
resume_from = None
work_dir = './work_dirs/ep3_r50_4dlongterm_depth_cbgs_debug'
