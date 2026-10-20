# G2 = G1 + heavy OCC convolution at 0.8 m (100x100).
# Upsample features, then predict at 0.4 m; GT/masks/range/Dz are unchanged.
_base_ = ['./bevdet-r50-det-occ-g1-occ128.py']
model = dict(
    occ_adapter=dict(dst_grid=[-40., 40., .8]),
    occ_head=dict(feature_upsample_scale=2, feature_align_corners=False))
work_dir = './work_dirs/ep1_g2_occ08'
