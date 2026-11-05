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

print('正式 RGB 数据 class_weight =', sem_class_weight)

default_scope = 'mmdet3d'

# 固定随机种子，保证不同实验尽量可比
randomness = dict(
    seed=42,
    deterministic=False
)


# ================= 3. 模型 =================
model = dict(
    type='S3DISOneFormer3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    in_channels=in_channels,
    num_channels=num_channels,
    voxel_size=0.01,
    num_classes=num_instance_classes,
    min_spatial_shape=128,

    # 创新2：GAASE shallow encoder
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True,
        gaase_stages=[1, 2],
        gaase_gate_hidden=16,
        gaase_iso_init_bias=1.5,
    ),

    # 创新1：D0 + D1 + D2 多尺度残差 Query 融合
    decoder=dict(
        type='QueryDecoder',
        num_layers=3,
        num_classes=num_instance_classes,
        num_instance_queries=400,
        num_semantic_queries=num_semantic_classes,
        num_instance_classes=num_instance_classes,
        in_channels=num_channels,
        multi_scale_in_channels=[64, 128, 192],
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


# ================= 4. 数据增强 =================
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

    # 保持与前面正式实验一致：关闭 ElasticTransfrom
    # dict(
    #     type='ElasticTransfrom',
    #     gran=[6, 20],
    #     mag=[40, 160],
    #     voxel_size=0.01,
    #     p=0.5),

    dict(
        type='Pack3DDetInputs_', _scope_='mmdet3d',
        keys=[
            'points',
            'gt_labels_3d',
            'pts_semantic_mask',
            'pts_instance_mask'])
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


# ================= 5. Dataloader =================
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


# ================= 6. Evaluator =================
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


# ================= 7. Optimizer =================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00005,
        weight_decay=0.05),
    clip_grad=dict(
        max_norm=10,
        norm_type=2))

param_scheduler = dict(
    type='PolyLR',
    begin=0,
    end=800,
    power=0.9)


# ================= 8. Hooks =================
custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=True)
]

default_hooks = dict(
    checkpoint=dict(
        interval=20,
        max_keep_ckpts=5,
        save_best=['pq', 'all_ap_50%', 'miou'],
        rule='greater'))


# ================= 9. 从头训练 =================
load_from = None

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=800,
    val_interval=20)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# ================= 10. Full model checkpoint selection rule =================
# 为保证与 Baseline / Innovation1 / Innovation2 公平比较：
# 统一采用 validation all_ap_50% 最优 checkpoint 作为正式结果。


# bestap-740 miou: 0.8279  all_ap: 0.8305  all_ap_50%: 0.9509  all_ap_25%: 0.9545  pq: 0.6608  data_time: 0.9972  time: 1.1290
# bestmiou-140 miou: 0.8144  all_ap: 0.7300  all_ap_50%: 0.8812  all_ap_25%: 0.9134  pq: 0.5998  data_time: 1.0000  time: 1.1047
# bestpq-300 miou: 0.8157  all_ap: 0.7661  all_ap_50%: 0.9246  all_ap_25%: 0.9378  pq: 0.6508  data_time: 1.0094  time: 1.0992

# 改val
# best-pq  miou: 0.8157  all_ap: 0.5891  all_ap_50%: 0.9623  all_ap_25%: 0.9689  pq: 0.6508  data_time: 0.9797  time: 1.0829
# best-miou miou: 0.8144  all_ap: 0.5741  all_ap_50%: 0.9053  all_ap_25%: 0.9567  pq: 0.5998  data_time: 0.9242  time: 1.0312
# best-ap  miou: 0.8279  all_ap: 0.6473  all_ap_50%: 0.9754  all_ap_25%: 0.9773  pq: 0.6608  data_time: 0.9126  time: 1.0205

# new
# best-miou  miou: 0.8390  all_ap: 0.5878  all_ap_50%: 0.9331  all_ap_25%: 0.9707  pq: 0.6569  data_time: 0.9273  time: 1.0604
# best-pq  miou: 0.8282  all_ap: 0.6417  all_ap_50%: 0.9562  all_ap_25%: 0.9768  pq: 0.6591  data_time: 1.0433  time: 1.1510
# best-ap  miou: 0.8281  all_ap: 0.6401  all_ap_50%: 0.9773  all_ap_25%: 0.9774  pq: 0.6746  data_time: 0.9280  time: 1.0347