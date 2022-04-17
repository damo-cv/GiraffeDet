_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
mid_ch=256
out_ch=256
model = dict(
    type='GFL',
    backbone=dict(
        type='Space2DepthChain',
    ),
    neck=dict(
        type='GiraffeNeck',
        min_level=3,
        max_level=7,
        num_levels=5,
        norm_layer=None,
        norm_kwargs=dict(eps=.001, momentum=.01),
        act_type='silu',
        fpn_config=None,
        fpn_name='giraffeneck',
        weight_method='concat',
        depth_multiplier=11,
        width_multiplier=0.7,
        with_backslash=True,
        with_slash=True,
        with_skip_connect=True,
        skip_connect_type='log2n',
        fpn_channels=[mid_ch, mid_ch, mid_ch, mid_ch, mid_ch],
        out_fpn_channels=[out_ch, out_ch, out_ch, out_ch, out_ch],
        pad_type='',
        downsample_type='max',
        upsample_type='nearest',
        apply_resample_bn=True,
        conv_after_downsample=False,
        redundant_bias=False,
        fpn_cell_repeats=1,
        separable_conv=False,
        conv_bn_relu_pattern=False,
        feature_info=[dict(num_chs=128, reduction=8), dict(num_chs=256, reduction=16), dict(num_chs=512, reduction=32), dict(num_chs=1024, reduction=64), dict(num_chs=2048, reduction=128)],
        alternate_init=False),
    bbox_head=dict(
        type='GFocalHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=False,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.250),
        reg_max=16,
        reg_topk=4,
        reg_channels=64,
        add_mean=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.6),
    max_per_img=100)
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=7330,
    warmup_ratio=0.1,
    step=[66, 71])

total_epochs = 73
runner = dict(type='EpochBasedRunner', max_epochs=73)

# multi-scale training
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 960)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=128),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(train=dict(pipeline=train_pipeline))


dataset_type = 'CocoDataset'
data_root = 'data/coco/'
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=128),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
