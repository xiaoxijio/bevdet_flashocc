# QAT fine-tuning of the completed four-GPU G5 fast run.
_base_ = ['../ep1_bevdet+flashocc/bevdet-b0-det-occ-g5-official-fast.py']
model = dict(occ_head=dict(use_conv_predicter=True))
qat = dict(
    enabled=True, algorithm='fake_quantization', chip_type='LQ560PlusV200',
    scopes=['img_backbone', 'img_neck', 'img_view_transformer.depth_net',
            'img_bev_encoder_backbone', 'img_bev_encoder_neck',
            'pts_bbox_head', 'occ_adapter.proj', 'occ_head'],
    calibration_max_batch=6, freeze_bn_epoch=0, freeze_observer_epoch=3, coalesced_sync=True)
load_from = './work_dirs/ep1_g5_official_fast/epoch_20_ema.pth'
resume_from = None
auto_resume = False
# Quantizer statistics must not be EMA averaged. QAT hook freezes BN itself.
custom_hooks = []
fp16 = None
# Four-GPU QAT throughput probes: batch 12 keeps FP32 fake quantization
# within memory limits. Keep 8 workers/GPU and about 4000 warmup samples.
data = dict(samples_per_gpu=12, workers_per_gpu=8)
optimizer = dict(lr=1e-5)
lr_config = dict(warmup_iters=84, step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)
work_dir = './work_dirs/ep2_g5_official_qat'
# QAT optimizer states are large; bound storage on the nearly full data disk.
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
