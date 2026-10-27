# Architecture/data recipe for subsequent Oritek QAT integration.
# This config alone does NOT enable QAT: tools/train.py has no QAT integration.
# G1: OCC width 128, no activation checkpointing or accelerated view transform.
# G3: shared BEV neck internal/output width 128 and deployment Resize mode.
# Keep OCC computation at 0.4 m (200x200), WITHOUT G2's 0.8 m / 2x upsample.
# Keep G2's training-only LiDAR depth supervision, weight 0.1.
_base_ = ['../ep1_bevdet+flashocc/bevdet-r50-det-occ-g1-occ128.py']

model = dict(
    depth_loss_weight=0.1,
    img_bev_encoder_neck=dict(out_channels=128, channels_factor=1,
                              align_corners=False),
    pts_bbox_head=dict(in_channels=128),
    occ_adapter=dict(in_channels=128, out_channels=128,
                     dst_grid=[-40., 40., 0.4]),
    occ_head=dict(in_dim=128, out_dim=128, feature_upsample_scale=1,
                  feature_align_corners=False))

# Floating-point initialization, not a QAT checkpoint or training resume.
# The G3 weights have matching parameter shapes; OCC sampling resolution changes.
load_from = './work_dirs/ep1_g3_bev128/epoch_20_ema.pth'
resume_from = None
work_dir = './work_dirs/ep2_bev128_occ04_depth_qat_base'

# LiDAR supplies training-only depth labels; inference stays camera-only.
# Load raw points after BEV augmentation for camera projection.
train_pipeline = [{'type': 'PrepareImageInputs',
  'is_train': True,
  'data_config': {'cams': ['CAM_FRONT_LEFT',
                           'CAM_FRONT',
                           'CAM_FRONT_RIGHT',
                           'CAM_BACK_LEFT',
                           'CAM_BACK',
                           'CAM_BACK_RIGHT'],
                  'Ncams': 6,
                  'input_size': (256, 704),
                  'src_size': (900, 1600),
                  'resize': (-0.06, 0.11),
                  'rot': (-5.4, 5.4),
                  'flip': True,
                  'crop_h': (0.0, 0.0),
                  'resize_test': 0.0}},
 {'type': 'LoadAnnotations'},
 {'type': 'LoadOccGTFromFile'},
 {'type': 'BEVAugDetOcc',
  'bda_aug_conf': {'rot_lim': (0.0, 0.0),
                   'scale_lim': (1.0, 1.0),
                   'flip_dx_ratio': 0.5,
                   'flip_dy_ratio': 0.5,
                   'tran_lim': (0.0, 0.0, 0.0)},
  'classes': ['car',
              'truck',
              'construction_vehicle',
              'bus',
              'trailer',
              'barrier',
              'motorcycle',
              'bicycle',
              'pedestrian',
              'traffic_cone']},
 {'type': 'LoadPointsFromFile',
  'coord_type': 'LIDAR',
  'load_dim': 5,
  'use_dim': 5,
  'file_client_args': {'backend': 'disk'}},
 {'type': 'PointToMultiViewDepth',
  'downsample': 1,
  'grid_config': {'x': [-51.2, 51.2, 0.8],
                  'y': [-51.2, 51.2, 0.8],
                  'z': [-5, 3, 8],
                  'depth': [1.0, 60.0, 1.0]}},
 {'type': 'ObjectRangeFilter',
  'point_cloud_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]},
 {'type': 'ObjectNameFilter',
  'classes': ['car',
              'truck',
              'construction_vehicle',
              'bus',
              'trailer',
              'barrier',
              'motorcycle',
              'bicycle',
              'pedestrian',
              'traffic_cone']},
 {'type': 'DefaultFormatBundle3D',
  'class_names': ['car',
                  'truck',
                  'construction_vehicle',
                  'bus',
                  'trailer',
                  'barrier',
                  'motorcycle',
                  'bicycle',
                  'pedestrian',
                  'traffic_cone']},
 {'type': 'FormatOccTargets'},
 {'type': 'Collect3D',
  'keys': ['img_inputs',
           'gt_bboxes_3d',
           'gt_labels_3d',
           'voxel_semantics',
           'mask_lidar',
           'mask_camera',
           'gt_depth']}]
data = dict(train=dict(dataset=dict(pipeline=train_pipeline)))
