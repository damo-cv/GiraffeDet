cudnn_benchmark = True
# model settings
# norm_cfg = dict(type='SyncBN', momentum=0.01, eps=1e-3, requires_grad=True)  # using SyncBN during training
norm_cfg = dict(type='BN', requires_grad=True)  # using SyncBN during training
model = dict(
    type='RetinaNet',
    pretrained='pretrained/adv-efficientnet-b0-b64d5a18.pth',
    backbone=dict(
        type='EfficientNet',
        arch='efficientnet-b0',
        out_indices=[4, 6, 8],
        norm_cfg=norm_cfg,
        norm_eval=True),
    neck=dict(
        type='BiFPN',
        in_channels=[40, 112, 320],
        target_size_list=[64, 32, 16, 8, 4],
        out_channels=64,
        stack=3,
        start_level=0,
        norm_cfg=norm_cfg,
        num_outs=5),
    bbox_head=dict(
        type='GFocalHead',
        num_classes=80,
        in_channels=64,
        stacked_convs=4,
        feat_channels=64,
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
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
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
    nms=dict(type='nms', iou_thr=0.6),
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_size = 512
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(img_size, img_size),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(img_size, img_size)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(img_size, img_size)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(img_size, img_size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(img_size, img_size)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
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
# evaluation = dict(interval=1, metric='bbox')
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.02,
    momentum=0.9,
    weight_decay=4e-5)
    # paramwise_cfg=dict(norm_decay_mult=0))
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.001,
    min_lr=0.0001)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 300
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/efficientdet_d0_gfocal'
load_from = None
resume_from = None
workflow = [('train', 1)]
