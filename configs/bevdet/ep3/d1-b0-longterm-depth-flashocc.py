# OCC width ablation: feature width 128, predictor hidden width 256.
# D1: B0 no-SE + BEVDepth4D longterm + joint 2D FlashOCC.
# 4 GPUs x 2 samples; 9 frames/sample. No autoscale-lr.
# Initialization: backbone extracted from ckpt/g5.pth; all other modules fresh.
# Folded BN hook follows G6 and freezes running statistics for ALL BN modules.
# Bilinear Resize align_corners=False targets ONNX pytorch_half_pixel; verify export.
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        sequential=True),
    dict(type='LoadAnnotations'),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='BEVAugDetOcc',
        bda_aug_conf=dict(
            rot_lim=(0.0, 0.0),
            scale_lim=(1.0, 1.0),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            tran_lim=(0.0, 0.0, 0.0)),
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='PointToMultiViewDepth',
        downsample=1,
        grid_config=dict(
            x=[-51.2, 51.2, 0.8],
            y=[-51.2, 51.2, 0.8],
            z=[-5, 3, 8],
            depth=[1.0, 60.0, 0.5])),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='CenterHeadTargetsCPU',
        train_cfg=dict(
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            grid_size=[1024, 1024, 40],
            voxel_size=[0.1, 0.1, 0.2],
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        tasks=[
            dict(
                num_class=10,
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ])
        ],
        norm_bbox=True,
        with_velocity=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(type='CropOccTargets', bounds=[36, 164]),
    dict(type='FormatOccTargets'),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_depth',
            'voxel_semantics', 'mask_lidar', 'mask_camera', 'det_targets'
        ])
]
test_pipeline = [
    dict(
        type='PrepareImageInputs',
        data_config=dict(
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        sequential=True),
    dict(type='LoadAnnotations'),
    dict(
        type='BEVAug',
        bda_aug_conf=dict(
            rot_lim=(0.0, 0.0),
            scale_lim=(1.0, 1.0),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            tran_lim=(0.0, 0.0, 0.0)),
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_infos_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        dataset=dict(
            data_root='data/nuscenes/',
            ann_file='data/nuscenes/bevdetv3-nuscenes_infos_train.pkl',
            pipeline=[
                dict(
                    type='PrepareImageInputs',
                    is_train=True,
                    data_config=dict(
                        cams=[
                            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.0),
                    sequential=True),
                dict(type='LoadAnnotations'),
                dict(type='LoadOccGTFromFile'),
                dict(
                    type='BEVAugDetOcc',
                    bda_aug_conf=dict(
                        rot_lim=(0.0, 0.0),
                        scale_lim=(1.0, 1.0),
                        flip_dx_ratio=0.5,
                        flip_dy_ratio=0.5,
                        tran_lim=(0.0, 0.0, 0.0)),
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='PointToMultiViewDepth',
                    downsample=1,
                    grid_config=dict(
                        x=[-51.2, 51.2, 0.8],
                        y=[-51.2, 51.2, 0.8],
                        z=[-5, 3, 8],
                        depth=[1.0, 60.0, 0.5])),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(
                    type='ObjectNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='CenterHeadTargetsCPU',
                    train_cfg=dict(
                        point_cloud_range=[
                            -51.2, -51.2, -5.0, 51.2, 51.2, 3.0
                        ],
                        grid_size=[1024, 1024, 40],
                        voxel_size=[0.1, 0.1, 0.2],
                        out_size_factor=8,
                        dense_reg=1,
                        gaussian_overlap=0.1,
                        max_objs=500,
                        min_radius=2,
                        code_weights=[
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
                        ]),
                    tasks=[
                        dict(
                            num_class=10,
                            class_names=[
                                'car', 'truck', 'construction_vehicle', 'bus',
                                'trailer', 'barrier', 'motorcycle', 'bicycle',
                                'pedestrian', 'traffic_cone'
                            ])
                    ],
                    norm_bbox=True,
                    with_velocity=True),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(type='CropOccTargets', bounds=[36, 164]),
                dict(type='FormatOccTargets'),
                dict(
                    type='Collect3D',
                    keys=[
                        'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d',
                        'gt_depth', 'voxel_semantics', 'mask_lidar',
                        'mask_camera', 'det_targets'
                    ])
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            test_mode=False,
            use_valid_flag=True,
            box_type_3d='LiDAR',
            type='NuScenesDatasetDetOcc',
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            img_info_prototype='bevdet4d',
            multi_adj_frame_id_cfg=(1, 9, 1),
            filter_empty_gt=False,
            occ_crop=[36, 164])),
    val=dict(
        type='NuScenesDatasetDetOcc',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/bevdetv3-nuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='PrepareImageInputs',
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                sequential=True),
            dict(type='LoadAnnotations'),
            dict(
                type='BEVAug',
                bda_aug_conf=dict(
                    rot_lim=(0.0, 0.0),
                    scale_lim=(1.0, 1.0),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5,
                    tran_lim=(0.0, 0.0, 0.0)),
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                is_train=False),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(type='Collect3D', keys=['points', 'img_inputs'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=(1, 9, 1),
        occ_crop=[36, 164]),
    test=dict(
        type='NuScenesDatasetDetOcc',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/bevdetv3-nuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='PrepareImageInputs',
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                sequential=True),
            dict(type='LoadAnnotations'),
            dict(
                type='BEVAug',
                bda_aug_conf=dict(
                    rot_lim=(0.0, 0.0),
                    scale_lim=(1.0, 1.0),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5,
                    tran_lim=(0.0, 0.0, 0.0)),
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                is_train=False),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(type='Collect3D', keys=['points', 'img_inputs'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=(1, 9, 1),
        occ_crop=[36, 164]))
evaluation = dict(
    interval=2,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ],
    metric=['bbox', 'occ'])
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ep3_d1_b0_longterm_depth_flashocc'
load_from = './ckpt/d1_b0_backbone_init.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
data_config = dict(
    cams=[
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0)
grid_config = dict(
    x=[-51.2, 51.2, 0.8],
    y=[-51.2, 51.2, 0.8],
    z=[-5, 3, 8],
    depth=[1.0, 60.0, 0.5])
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 80
multi_adj_frame_id_cfg = (1, 9, 1)
model = dict(
    type='BEVDepth4DDetOcc',
    align_after_view_transfromation=False,
    num_adj=8,
    img_backbone=dict(
        type='EfficientNetB0HZ', pretrained=None, norm_eval=True),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[112, 1280],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0],
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=dict(
            x=[-51.2, 51.2, 0.8],
            y=[-51.2, 51.2, 0.8],
            z=[-5, 3, 8],
            depth=[1.0, 60.0, 0.5]),
        input_size=(256, 704),
        in_channels=256,
        out_channels=64,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet', numC_input=576, num_channels=[128, 256, 512]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=640,
        out_channels=128,
        channels_factor=1,
        align_corners=False),
    pre_process=dict(
        type='CustomResNet',
        numC_input=64,
        num_layer=[2],
        num_channels=[64],
        stride=[1],
        backbone_output_ids=[0]),
    pts_bbox_head=dict(
        type='CenterHeadCPUTargets',
        in_channels=128,
        tasks=[
            dict(
                num_class=10,
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=[-51.2, -51.2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(
            type='GaussianFocalLoss', reduction='mean', loss_weight=6.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=1.5),
        norm_bbox=True),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            grid_size=[1024, 1024, 40],
            voxel_size=[0.1, 0.1, 0.2],
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=500,
            nms_type=['rotate'],
            nms_thr=[0.2],
            nms_rescale_factor=[[
                1.0, 0.7, 0.7, 0.4, 0.55, 1.1, 1.0, 1.0, 1.5, 3.5
            ]])),
    with_prev=True,
    occ_adapter=dict(
        type='BEVGridAdapter',
        src_grid=[-51.2, 51.2, 0.8],
        dst_grid=[-25.6, 25.6, 0.4],
        in_channels=128,
        out_channels=128),
    occ_head=dict(
        type='BEVOCCHead2D',
        in_dim=128,
        out_dim=128,
        predictor_hidden_dim=256,
        feature_upsample_scale=1,
        Dz=16,
        use_mask=True,
        num_classes=18,
        use_predicter=True,
        class_balance=False,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            avg_non_ignore=True,
            ignore_index=255,
            loss_weight=1.0)),
    occ_loss_weight=1.0,
    depth_loss_weight=3.0,
    depth_loss_from_logits=False)
bda_aug_conf = dict(
    rot_lim=(0.0, 0.0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    tran_lim=(0.0, 0.0, 0.0))
share_data_config = dict(
    type='NuScenesDataset',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=(1, 9, 1))
test_data_config = dict(
    pipeline=[
        dict(
            type='PrepareImageInputs',
            data_config=dict(
                cams=[
                    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                ],
                Ncams=6,
                input_size=(256, 704),
                src_size=(900, 1600),
                resize=(-0.06, 0.11),
                rot=(-5.4, 5.4),
                flip=True,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            sequential=True),
        dict(type='LoadAnnotations'),
        dict(
            type='BEVAug',
            bda_aug_conf=dict(
                rot_lim=(-22.5, 22.5),
                scale_lim=(0.95, 1.05),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5),
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            is_train=False),
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(type='Collect3D', keys=['points', 'img_inputs'])
            ])
    ],
    ann_file='data/nuscenes/bevdetv3-nuscenes_infos_val.pkl',
    type='NuScenesDataset',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=(1, 9, 1))
key = 'test'
optimizer = dict(type='AdamW', lr=5e-05, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)
custom_hooks = [
    dict(type='G5FoldedBatchNormHook', priority='HIGH'),
    dict(type='MEGVIIEMAHook', init_updates=0, priority='NORMAL')
]
custom_imports = dict(
    imports=[
        'mmdet3d.models.backbones.efficientnet_b0_hz',
        'mmdet3d.models.dense_heads.center_head_cpu_targets',
        'mmdet3d.models.detectors.bevdepth4d_det_occ',
        'mmdet3d.datasets.pipelines.center_head_targets_cpu',
        'mmdet3d.datasets.pipelines.crop_occ_targets'
    ],
    allow_failed_imports=False)
auto_resume = False
fp16 = dict(loss_scale=dict(init_scale=512.0, growth_interval=2000))
find_unused_parameters = False
