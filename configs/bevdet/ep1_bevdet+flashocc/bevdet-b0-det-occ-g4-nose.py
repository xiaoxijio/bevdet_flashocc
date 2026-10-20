# G4 = G3 + torchvision EfficientNetB0 without SE, with ReLU6.
# Not the SDK EfficientNetB0HZ implementation/checkpoint format.
_base_ = ['./bevdet-r50-det-occ-g3-bev128.py']
model = dict(
    img_backbone=dict(
        _delete_=True, type='EfficientNetB0NoSE', norm_eval=False,
        # Retained ImageNet B0 weights initialize Conv/BN layers. SE is removed
        # and SiLU is replaced, so task fine-tuning remains necessary.
        pretrained='https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth'),
    img_neck=dict(in_channels=[112, 1280],
                  upsample_cfg=dict(mode='bilinear', align_corners=False)))
# The teacher's ResNet keys do not match this new features.* backbone.
# Other matching teacher tensors are still restored by load_from.
work_dir = './work_dirs/ep1_g4_b0_nose'
