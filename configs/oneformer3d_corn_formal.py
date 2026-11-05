_base_ = [
    'mmdet3d::_base_/default_runtime.py',
]
custom_imports = dict(imports=['oneformer3d'])

# ================= 1. 核心类别与维度设置 =================
# 与 HAIS 严格对齐：2 个类 (0 和 1)
num_instance_classes = 2
num_semantic_classes = 2

# 与 HAIS 严格对齐：只有 XYZ 三个通道
in_channels = 3
num_channels = 64

model = dict(
    type='S3DISOneFormer3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    in_channels=in_channels,
    num_channels=num_channels,
    # voxel_size=0.02,
    voxel_size=0.01,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    decoder=dict(
        type='QueryDecoder',
        num_layers=3,
        num_classes=num_instance_classes,
        num_instance_queries=400,
        num_semantic_queries=num_semantic_classes,
        num_instance_classes=num_instance_classes,
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=True),
    criterion=dict(
        type='S3DISUnifiedCriterion',
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type='S3DISSemanticCriterion',
            loss_weight=5.0,
            seg_loss=dict(
        type='mmdet.CrossEntropyLoss',
        use_sigmoid=True,
        class_weight=[3.29, 0.59])),
        inst_criterion=dict(
            type='InstanceCriterion',
            matcher=dict(
                type='HungarianMatcher',
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0)]),
            loss_weight=[0.5, 1.0, 1.0, 0.5],
            num_classes=num_instance_classes,
            non_object_weight=0.05,
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=600,
        inst_score_thr=0.0,
        pan_score_thr=0.4,
        npoint_thr=50, # 回退到 HAIS 的 TEST_NPOINT_THRESH: 300
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.15, # 与 HAIS 的 TEST_SCORE_THRESH 完全一致
        nms=True,
        matrix_nms_kernel='linear',
        num_sem_cls=num_semantic_classes,
        stuff_cls=[],
        thing_cls=[0, 1]
    )
)

# ================= 2. 数据流与增强管道 =================
dataset_type = 'CornSegDataset'
data_root = 'D:/onef/2time/oneformer/'

train_pipeline = [
    dict(
        type='LoadNpyPointsFromFile', _scope_='mmdet3d',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2]),

    dict(type='PointInstClassMapping_', _scope_='mmdet3d',
        num_classes=num_instance_classes),

    dict(
        type='RandomFlip3D', _scope_='mmdet3d',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),

    dict(
        type='GlobalRotScaleTrans', _scope_='mmdet3d',
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),

    dict(
        type='RandomJitterPoints', _scope_='mmdet3d',
        jitter_std=[0.005, 0.005, 0.005],
        clip_range=[-0.02, 0.02]),

    dict(type='PointShuffle', _scope_='mmdet3d'),

    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=0.01,
        p=0.5),

    dict(
        type='Pack3DDetInputs_', _scope_='mmdet3d',
        keys=['points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask'])
]

test_pipeline = [
    dict(
        type='LoadNpyPointsFromFile', _scope_='mmdet3d',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(type='Pack3DDetInputs_', _scope_='mmdet3d', keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
]

# ================= 3. 数据加载器 =================
train_dataloader = dict(
    batch_size=2, # OneFormer3D 比 HAIS 沉重，建议先用 2。如果训练时通过 `nvidia-smi` 发现显存占用不多，再改成 4
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='corn_infos_train.pkl',
        pipeline=train_pipeline,
        filter_empty_gt=False,
        box_type_3d='Depth',
        backend_args=None))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='corn_infos_val.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        box_type_3d='Depth',
        backend_args=None))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='corn_infos_test.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        box_type_3d='Depth',
        backend_args=None))

# ================= 4. 评估指标 =================
class_names = ['stem', 'leaf']
label2cat = {i: name for i, name in enumerate(class_names)}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[],        # 2是unlabeled（实际不存在，不影响结果）
    classes=class_names,
    dataset_name='S3DIS')
sem_mapping = [0, 1]
inst_mapping = [0, 1]

val_evaluator = dict(
    type='UnifiedSegMetric',
    stuff_class_inds=[],
    thing_class_inds=[0, 1],
    min_num_points=1,
    id_offset=2**16,
    sem_mapping=sem_mapping,
    # inst_mapping=sem_mapping,
    inst_mapping=inst_mapping,
    submission_prefix_semantic=None,
    submission_prefix_instance=None,
    metric_meta=metric_meta)
test_evaluator = val_evaluator

# ================= 5. 优化器与训练周期 =================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00005, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = dict(type='PolyLR', begin=0, end=800, power=0.9)



custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
default_hooks = dict(
    checkpoint=dict(
        interval=10,
        # max_keep_ckpts=3,
        max_keep_ckpts=5,
        save_best=['all_ap_50%', 'miou'],
        rule='greater'))

# 注释掉了原版的预训练权重，我们直接从头炼丹
# load_from = 'work_dirs/tmp/instance-only-oneformer3d_1xb2_scannet-and-structured3d.pth'
# load_from = 'work_dirs/oneformer3d_corn_win/best_all_ap_50%_epoch_800.pth'
# [重点对齐] 严格按照 HAIS 设置 120 轮
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=800, val_interval=20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# miou: 0.6639  all_ap: 0.5062  all_ap_50%: 0.7102  all_ap_25%: 0.8019  pq: 0.4184  data_time: 0.9900  time: 1.1182  660-miou
# miou: 0.6757  all_ap: 0.4936  all_ap_50%: 0.6976  all_ap_25%: 0.7951  pq: 0.4078  data_time: 0.9711  time: 1.0883  700-ap50