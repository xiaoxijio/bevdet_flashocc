# G3 = G2 + shared neck internal/output width 128, deployment Resize mode.
# The 640-channel concatenated neck input and BEV backbone are unchanged.
_base_ = ['./bevdet-r50-det-occ-g2-occ08.py']
model = dict(
    img_bev_encoder_neck=dict(out_channels=128, channels_factor=1,
                              align_corners=False),
    pts_bbox_head=dict(in_channels=128),
    occ_adapter=dict(in_channels=128))
work_dir = './work_dirs/ep1_g3_bev128'
