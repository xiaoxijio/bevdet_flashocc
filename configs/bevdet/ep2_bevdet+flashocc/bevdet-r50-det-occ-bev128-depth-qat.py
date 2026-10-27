# INT8 QAT on convolution islands; runtime mapping/geometry/losses remain FP.
_base_ = ['./bevdet-r50-det-occ-bev128-depth-qat-base.py']
model = dict(occ_head=dict(use_conv_predicter=True))
qat = dict(
    enabled=True,
    algorithm='fake_quantization',
    chip_type='LQ560PlusV200',
    scopes=['img_backbone', 'img_neck', 'img_view_transformer.depth_net',
            'img_bev_encoder_backbone', 'img_bev_encoder_neck',
            'pts_bbox_head', 'occ_adapter.proj', 'occ_head'],
    calibration_max_batch=6,
    # Freeze pretrained BN immediately, collectors after three full epochs.
    freeze_bn_epoch=0,
    freeze_observer_epoch=3)
# Existing Linear weights are reshaped to equivalent 1x1 Conv2d on FP load.
load_from = './work_dirs/ep1_g3_bev128/epoch_20_ema.pth'
resume_from = None
auto_resume = False
custom_hooks = []  # Do not EMA-average scales/zero points/min/max.
fp16 = None
# Start conservatively; validate task metrics before extending the QAT run.
optimizer = dict(lr=1e-5)
lr_config = dict(warmup_iters=100, step=[6, 9])
runner = dict(type='EpochBasedRunner', max_epochs=10)
work_dir = './work_dirs/ep2_bev128_occ04_depth_qat'
