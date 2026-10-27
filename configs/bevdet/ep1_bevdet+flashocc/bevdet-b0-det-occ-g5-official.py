# SDK official optimized detection architecture + the existing joint OCC branch.
# This reproduces the official exported detection function at initialization;
# the guide does not publish a complete joint Det/OCC training recipe.
_base_ = ['./bevdet-r50-det-occ-g1-occ128.py']
custom_imports = dict(
    imports=['mmdet3d.models.backbones.efficientnet_b0_hz'],
    allow_failed_imports=False)

model = dict(
    img_backbone=dict(_delete_=True, type='EfficientNetB0HZ',
                      pretrained=None, norm_eval=True),
    img_neck=dict(in_channels=[112, 1280],
                  upsample_cfg=dict(mode='bilinear', align_corners=False)),
    img_bev_encoder_neck=dict(out_channels=128, channels_factor=1,
                              align_corners=False),
    pts_bbox_head=dict(in_channels=128),
    occ_adapter=dict(in_channels=128, out_channels=128,
                     dst_grid=[-40., 40., .4]),
    occ_head=dict(in_dim=128, out_dim=128, feature_upsample_scale=1),
    # The official BEVDet has no auxiliary LiDAR depth loss. OCC is our extension.
    depth_loss_weight=0., depth_loss_from_logits=False, transfer_from=None)

# Generated from the official trained ONNX + matching G3 OCC tensors.
# Never substitute the G4 checkpoint or torchvision ImageNet B0 weights.
load_from = './ckpt/g5_official_det_g3_occ_init.pth'
resume_from = None
auto_resume = False
custom_hooks = [
    dict(type='G5FoldedBatchNormHook', priority='HIGH'),
    dict(type='MEGVIIEMAHook', init_updates=0, priority='NORMAL')]
# Joint fine-tuning schedule (not an undocumented claim about SDK pretraining).
data = dict(samples_per_gpu=2, workers_per_gpu=4)
optimizer = dict(lr=5e-5)
lr_config = dict(warmup_iters=500, step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)
work_dir = './work_dirs/ep1_g5_official'
