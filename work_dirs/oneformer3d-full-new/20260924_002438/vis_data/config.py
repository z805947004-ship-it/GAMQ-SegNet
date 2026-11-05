class_names = [
    'stem',
    'leaf',
]
custom_hooks = [
    dict(after_iter=True, type='EmptyCacheHook'),
]
custom_imports = dict(imports=[
    'oneformer3d',
])
data_root = 'D:/onef/2time/oneformer_rgb/'
dataset_type = 'CornSegDataset'
default_hooks = dict(
    checkpoint=dict(
        _scope_='mmdet3d',
        interval=20,
        max_keep_ckpts=5,
        rule='greater',
        save_best=[
            'pq',
            'all_ap_50%',
            'miou',
        ],
        type='CheckpointHook'),
    logger=dict(_scope_='mmdet3d', interval=50, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet3d', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet3d', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet3d', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet3d', type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
in_channels = 6
inst_mapping = [
    0,
    1,
]
label2cat = dict({0: 'stem', 1: 'leaf'})
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet3d', by_epoch=True, type='LogProcessor', window_size=50)
metric_meta = dict(
    classes=[
        'stem',
        'leaf',
    ],
    dataset_name='S3DIS',
    ignore_index=[],
    label2cat=dict({
        0: 'stem',
        1: 'leaf'
    }))
model = dict(
    backbone=dict(
        gaase_gate_hidden=16,
        gaase_iso_init_bias=1.5,
        gaase_stages=[
            1,
            2,
        ],
        num_planes=[
            64,
            128,
            192,
            256,
            320,
        ],
        return_blocks=True,
        type='SpConvUNet'),
    criterion=dict(
        inst_criterion=dict(
            fix_dice_loss_weight=True,
            fix_mean_loss=True,
            iter_matcher=True,
            loss_weight=[
                0.5,
                1.0,
                1.0,
                0.5,
            ],
            matcher=dict(
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0),
                ],
                type='HungarianMatcher'),
            non_object_weight=0.05,
            num_classes=2,
            type='InstanceCriterion'),
        num_semantic_classes=2,
        sem_criterion=dict(
            loss_weight=5.0,
            seg_loss=dict(
                class_weight=[
                    4.2863,
                    0.5894,
                ],
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True),
            type='S3DISSemanticCriterion'),
        type='S3DISUnifiedCriterion'),
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    decoder=dict(
        activation_fn='gelu',
        attn_mask=True,
        d_model=256,
        dropout=0.0,
        fix_attention=True,
        hidden_dim=1024,
        in_channels=64,
        iter_pred=True,
        multi_scale_in_channels=[
            64,
            128,
            192,
        ],
        num_classes=2,
        num_heads=8,
        num_instance_classes=2,
        num_instance_queries=400,
        num_layers=3,
        num_semantic_queries=2,
        objectness_flag=True,
        type='QueryDecoder'),
    in_channels=6,
    min_spatial_shape=128,
    num_channels=64,
    num_classes=2,
    test_cfg=dict(
        inst_score_thr=0.0,
        matrix_nms_kernel='linear',
        nms=True,
        npoint_thr=50,
        num_sem_cls=2,
        obj_normalization=True,
        obj_normalization_thr=0.01,
        pan_score_thr=0.4,
        sp_score_thr=0.15,
        stuff_cls=[],
        thing_cls=[
            0,
            1,
        ],
        topk_insts=600),
    train_cfg=dict(),
    type='S3DISOneFormer3D',
    voxel_size=0.01)
num_channels = 64
num_instance_classes = 2
num_semantic_classes = 2
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(lr=5e-05, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')
param_scheduler = dict(begin=0, end=800, power=0.9, type='PolyLR')
randomness = dict(deterministic=False, seed=42)
resume = False
sem_class_weight = [
    4.2863,
    0.5894,
]
sem_mapping = [
    0,
    1,
]
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='corn_infos_test.pkl',
        backend_args=None,
        box_type_3d='Depth',
        data_root='D:/onef/2time/oneformer_rgb/',
        pipeline=[
            dict(
                _scope_='mmdet3d',
                coord_type='DEPTH',
                load_dim=6,
                shift_height=False,
                type='LoadNpyPointsFromFile',
                use_color=True,
                use_dim=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                ]),
            dict(
                _scope_='mmdet3d',
                keys=[
                    'points',
                    'pts_semantic_mask',
                    'pts_instance_mask',
                ],
                type='Pack3DDetInputs_'),
        ],
        test_mode=True,
        type='CornSegDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    id_offset=65536,
    inst_mapping=[
        0,
        1,
    ],
    metric_meta=dict(
        classes=[
            'stem',
            'leaf',
        ],
        dataset_name='S3DIS',
        ignore_index=[],
        label2cat=dict({
            0: 'stem',
            1: 'leaf'
        })),
    min_num_points=1,
    sem_mapping=[
        0,
        1,
    ],
    stuff_class_inds=[],
    submission_prefix_instance=None,
    submission_prefix_semantic=None,
    thing_class_inds=[
        0,
        1,
    ],
    type='UnifiedSegMetric')
test_pipeline = [
    dict(
        _scope_='mmdet3d',
        coord_type='DEPTH',
        load_dim=6,
        shift_height=False,
        type='LoadNpyPointsFromFile',
        use_color=True,
        use_dim=[
            0,
            1,
            2,
            3,
            4,
            5,
        ]),
    dict(
        _scope_='mmdet3d',
        keys=[
            'points',
            'pts_semantic_mask',
            'pts_instance_mask',
        ],
        type='Pack3DDetInputs_'),
]
train_cfg = dict(max_epochs=800, type='EpochBasedTrainLoop', val_interval=20)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='corn_infos_train.pkl',
        backend_args=None,
        box_type_3d='Depth',
        data_root='D:/onef/2time/oneformer_rgb/',
        filter_empty_gt=False,
        pipeline=[
            dict(
                _scope_='mmdet3d',
                coord_type='DEPTH',
                load_dim=6,
                shift_height=False,
                type='LoadNpyPointsFromFile',
                use_color=True,
                use_dim=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                ]),
            dict(
                _scope_='mmdet3d',
                num_classes=2,
                type='PointInstClassMapping_'),
            dict(
                _scope_='mmdet3d',
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5,
                sync_2d=False,
                type='RandomFlip3D'),
            dict(
                _scope_='mmdet3d',
                rot_range=[
                    -3.14,
                    3.14,
                ],
                scale_ratio_range=[
                    0.9,
                    1.1,
                ],
                shift_height=False,
                translation_std=[
                    0.03,
                    0.03,
                    0.03,
                ],
                type='GlobalRotScaleTrans'),
            dict(
                _scope_='mmdet3d',
                clip_range=[
                    -0.01,
                    0.01,
                ],
                jitter_std=[
                    0.002,
                    0.002,
                    0.002,
                ],
                type='RandomJitterPoints'),
            dict(_scope_='mmdet3d', type='PointShuffle'),
            dict(
                _scope_='mmdet3d',
                keys=[
                    'points',
                    'gt_labels_3d',
                    'pts_semantic_mask',
                    'pts_instance_mask',
                ],
                type='Pack3DDetInputs_'),
        ],
        type='CornSegDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        _scope_='mmdet3d',
        coord_type='DEPTH',
        load_dim=6,
        shift_height=False,
        type='LoadNpyPointsFromFile',
        use_color=True,
        use_dim=[
            0,
            1,
            2,
            3,
            4,
            5,
        ]),
    dict(_scope_='mmdet3d', num_classes=2, type='PointInstClassMapping_'),
    dict(
        _scope_='mmdet3d',
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        sync_2d=False,
        type='RandomFlip3D'),
    dict(
        _scope_='mmdet3d',
        rot_range=[
            -3.14,
            3.14,
        ],
        scale_ratio_range=[
            0.9,
            1.1,
        ],
        shift_height=False,
        translation_std=[
            0.03,
            0.03,
            0.03,
        ],
        type='GlobalRotScaleTrans'),
    dict(
        _scope_='mmdet3d',
        clip_range=[
            -0.01,
            0.01,
        ],
        jitter_std=[
            0.002,
            0.002,
            0.002,
        ],
        type='RandomJitterPoints'),
    dict(_scope_='mmdet3d', type='PointShuffle'),
    dict(
        _scope_='mmdet3d',
        keys=[
            'points',
            'gt_labels_3d',
            'pts_semantic_mask',
            'pts_instance_mask',
        ],
        type='Pack3DDetInputs_'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='corn_infos_val.pkl',
        backend_args=None,
        box_type_3d='Depth',
        data_root='D:/onef/2time/oneformer_rgb/',
        pipeline=[
            dict(
                _scope_='mmdet3d',
                coord_type='DEPTH',
                load_dim=6,
                shift_height=False,
                type='LoadNpyPointsFromFile',
                use_color=True,
                use_dim=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                ]),
            dict(
                _scope_='mmdet3d',
                keys=[
                    'points',
                    'pts_semantic_mask',
                    'pts_instance_mask',
                ],
                type='Pack3DDetInputs_'),
        ],
        test_mode=True,
        type='CornSegDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    id_offset=65536,
    inst_mapping=[
        0,
        1,
    ],
    metric_meta=dict(
        classes=[
            'stem',
            'leaf',
        ],
        dataset_name='S3DIS',
        ignore_index=[],
        label2cat=dict({
            0: 'stem',
            1: 'leaf'
        })),
    min_num_points=1,
    sem_mapping=[
        0,
        1,
    ],
    stuff_class_inds=[],
    submission_prefix_instance=None,
    submission_prefix_semantic=None,
    thing_class_inds=[
        0,
        1,
    ],
    type='UnifiedSegMetric')
work_dir = 'work_dirs/oneformer3d-full-new'
