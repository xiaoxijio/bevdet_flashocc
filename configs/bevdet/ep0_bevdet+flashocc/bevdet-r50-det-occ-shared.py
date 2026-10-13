# Single-frame BEVDet + FlashOCC with ONE shared BEV encoder.
_base_ = ['../bevdet-r50-cbgs.py']

class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
               'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
point_cloud_range = [-51.2, -51.2, -5., 51.2, 51.2, 3.]
data_root = 'data/nuscenes/'
data_config = dict(
    cams=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
          'CAM_BACK', 'CAM_BACK_RIGHT'], Ncams=6,
    input_size=(256, 704), src_size=(900, 1600), resize=(-.06, .11),
    rot=(-5.4, 5.4), flip=True, crop_h=(0., 0.), resize_test=0.)

model = dict(
    type='BEVDetDetOcc',
    occ_adapter=dict(type='BEVGridAdapter', src_grid=[-51.2, 51.2, .8],
                     dst_grid=[-40., 40., .4], in_channels=256, out_channels=128),
    occ_head=dict(type='BEVOCCHead2D', in_dim=128, out_dim=128, Dz=16,
                  use_mask=True, num_classes=18, use_predicter=True,
                  class_balance=False,
                  loss_occ=dict(type='CrossEntropyLoss', use_sigmoid=False,
                                ignore_index=255, loss_weight=1.)),
    occ_loss_weight=1.)

# Existing occupancy augmentation handles flips, not voxel rotations/scaling.
bda_aug_conf = dict(rot_lim=(0., 0.), scale_lim=(1., 1.),
                    flip_dx_ratio=.5, flip_dy_ratio=.5, tran_lim=(0., 0., 0.))
train_pipeline = [
    dict(type='PrepareImageInputs', is_train=True, data_config=data_config),
    dict(type='LoadAnnotations'),
    dict(type='LoadOccGTFromFile'),
    dict(type='BEVAugDetOcc', bda_aug_conf=bda_aug_conf, classes=class_names),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='FormatOccTargets'),
    dict(type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d',
                               'voxel_semantics', 'mask_lidar', 'mask_camera'])]

data = dict(
    samples_per_gpu=2, workers_per_gpu=4,
    train=dict(dataset=dict(type='NuScenesDatasetDetOcc', pipeline=train_pipeline,
                            filter_empty_gt=False)),
    val=dict(type='NuScenesDatasetDetOcc'),
    test=dict(type='NuScenesDatasetDetOcc'))

# Starting single-GPU recipe: base batch 8 -> batch 2; retune for GPU count.
optimizer = dict(lr=5e-5)
lr_config = dict(step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)
evaluation = dict(interval=1, metric=['bbox', 'occ'])
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
work_dir = './work_dirs/bevdet-r50-det-occ-shared'
