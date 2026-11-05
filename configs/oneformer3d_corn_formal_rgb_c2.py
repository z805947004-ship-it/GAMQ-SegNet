_base_ = [
    'mmdet3d::_base_/default_runtime.py',
]
custom_imports = dict(imports=['oneformer3d'])


# ================= 1. 数据路径 =================
dataset_type = 'CornSegDataset'
data_root = 'D:/onef/2time/oneformer_rgb/'


# ================= 2. 类别与输入维度 =================
num_instance_classes = 2
num_semantic_classes = 2

# N×6: x y z r g b
in_channels = 6
num_channels = 64

class_names = ['stem', 'leaf']
label2cat = {i: name for i, name in enumerate(class_names)}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[],
    classes=class_names,
    dataset_name='S3DIS')

sem_mapping = [0, 1]
inst_mapping = [0, 1]

sem_class_weight = [4.2863, 0.5894]
# ================= 3. 自动计算 class_weight =================

print('正式 RGB 数据 class_weight =', sem_class_weight)

default_scope = 'mmdet3d'
# 固定随机种子，保证不同实验尽量可比
randomness = dict(
    seed=42,
    deterministic=False
)
# ================= 4. 模型 =================
model = dict(
    type='S3DISOneFormer3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    in_channels=in_channels,
    num_channels=num_channels,
    voxel_size=0.01,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True,
        gaase_stages=[1, 2],
        gaase_gate_hidden=16,
        gaase_iso_init_bias=1.5,
    ),
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
                class_weight=sem_class_weight)),
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
        npoint_thr=50,
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.15,
        nms=True,
        matrix_nms_kernel='linear',
        num_sem_cls=num_semantic_classes,
        stuff_cls=[],
        thing_cls=[0, 1])
)


# ================= 5. 数据增强 =================
train_pipeline = [
    dict(
        type='LoadNpyPointsFromFile', _scope_='mmdet3d',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),

    # 不再使用 PointSample，保留每个点云真实点数

    dict(
        type='PointInstClassMapping_', _scope_='mmdet3d',
        num_classes=num_instance_classes),

    dict(
        type='RandomFlip3D', _scope_='mmdet3d',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),

    dict(
        type='GlobalRotScaleTrans', _scope_='mmdet3d',
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.03, 0.03, 0.03],
        shift_height=False),

    dict(
        type='RandomJitterPoints', _scope_='mmdet3d',
        jitter_std=[0.002, 0.002, 0.002],
        clip_range=[-0.01, 0.01]),

    dict(type='PointShuffle', _scope_='mmdet3d'),

    # 先关闭 ElasticTransfrom。旧结果实例粘连明显，先避免扰乱细长叶片边界。
    # dict(
    #     type='ElasticTransfrom',
    #     gran=[6, 20],
    #     mag=[40, 160],
    #     voxel_size=0.01,
    #     p=0.5),

    dict(
        type='Pack3DDetInputs_', _scope_='mmdet3d',
        keys=['points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask'])
]

test_pipeline = [
    dict(
        type='LoadNpyPointsFromFile', _scope_='mmdet3d',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),

    dict(
        type='Pack3DDetInputs_', _scope_='mmdet3d',
        keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
]


# ================= 6. Dataloader =================
train_dataloader = dict(
    batch_size=2,
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


# ================= 7. Evaluator =================
val_evaluator = dict(
    type='UnifiedSegMetric',
    stuff_class_inds=[],
    thing_class_inds=[0, 1],
    min_num_points=1,
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=inst_mapping,
    submission_prefix_semantic=None,
    submission_prefix_instance=None,
    metric_meta=metric_meta)

test_evaluator = val_evaluator


# ================= 8. Optimizer =================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00005, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = dict(type='PolyLR', begin=0, end=800, power=0.9)


# ================= 9. Hooks =================
custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=True)
]

default_hooks = dict(
    checkpoint=dict(
        interval=20,
        max_keep_ckpts=5,
        save_best=['pq', 'all_ap_50%', 'miou'],
        rule='greater'))


# ================= 10. 从头训练 =================
load_from = None

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=800, val_interval=20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# bestpq-760  miou: 0.8478  all_ap: 0.8334  all_ap_50%: 0.9472  all_ap_25%: 0.9692  pq: 0.6834  data_time: 1.0690  time: 1.2010
# bestmiou-140 miou: 0.8165  all_ap: 0.7206  all_ap_50%: 0.8655  all_ap_25%: 0.9237  pq: 0.6303  data_time: 1.0263  time: 1.1217
# bestap-420  miou: 0.8340  all_ap: 0.8071  all_ap_50%: 0.9334  all_ap_25%: 0.9339  pq: 0.6296  data_time: 1.0396  time: 1.1300

# 改val
# bestpq  miou: 0.8478  all_ap: 0.6962  all_ap_50%: 0.9736  all_ap_25%: 0.9846  pq: 0.6834  data_time: 0.9274  time: 1.0222
# bestmiou miou: 0.8165  all_ap: 0.5924  all_ap_50%: 0.8829  all_ap_25%: 0.9618  pq: 0.6303  data_time: 0.9416  time: 1.0245
# bestap miou: 0.8340  all_ap: 0.6279  all_ap_50%: 0.8865  all_ap_25%: 0.9670  pq: 0.6296  data_time: 0.9479  time: 1.0500
