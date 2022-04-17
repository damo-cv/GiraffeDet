model = dict(
    type='SingleStageDetector',
    backbone=dict(type='DarknetCSP', scale='v4l5p', out_indices=[3, 4, 5]),
    neck=dict(
        type='YOLOV4Neck',
        in_channels=[256, 512, 512],
        out_channels=[256, 512, 1024],
        csp_repetition=2),
    bbox_head=dict(
        type='YOLOCSPHead', num_classes=80, in_channels=[256, 512, 1024]),
    train_cfg=dict(),
    test_cfg=dict(
        min_bbox_size=0,
        nms_pre=-1,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300))

dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(mean=[114, 114, 114], std=[255, 255, 255], to_rgb=True)
train_pipeline = [
    dict(
        type='MosaicPipeline',
        individual_pipeline=[
            dict(type='LoadImageFromFile', im_decode_backend='turbojpeg'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
        ],
        pad_val=114),
    dict(
        type='Albu',
        update_pad_shape=True,
        skip_img_without_anno=False,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            min_area=4,
            min_visibility=0.2,
            label_fields=['gt_labels'],
            check_each_transform=False),
        transforms=[
            dict(
                type='PadIfNeeded',
                min_height=1920,
                min_width=1920,
                border_mode=0,
                value=(114, 114, 114),
                always_apply=True),
            dict(
                type='RandomCrop', width=1280, height=1280, always_apply=True),
            dict(
                type='RandomScale',
                scale_limit=0.5,
                interpolation=1,
                always_apply=True),
            dict(type='CenterCrop', width=640, height=640, always_apply=True),
            dict(type='HorizontalFlip', p=0.5)
        ]),
    dict(
        type='HueSaturationValueJitter',
        hue_ratio=0.015,
        saturation_ratio=0.7,
        value_ratio=0.4),
    dict(type='GtBBoxesFilter', min_size=2, max_aspect_ratio=20),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        samples_per_gpu=8,
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

nominal_batch_size = 64

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.))

optimizer_config = dict(
    type='Fp16GradAccumulateOptimizerHook',
    nominal_batch_size=nominal_batch_size,
    grad_clip=dict(max_norm=35, norm_type=2),
    loss_scale='dynamic')

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.2,
)

load_from = 'work_dirs/tencent_traffic_sign_yolov4l/latest.pth'
resume_from = 'work_dirs/tencent_traffic_sign_yolov4l/latest.pth'

custom_hooks = [
    dict(
        type='DetailedLinearWarmUpHook',
        warmup_iters=10000,
        lr_weight_warmup_ratio=0.,
        lr_bias_warmup_ratio=10.,
        momentum_warmup_ratio=0.95,
        priority='NORMAL'),
    dict(
        type='StateEMAHook',
        momentum=0.9999,
        nominal_batch_size=nominal_batch_size,
        warm_up=10000,
        resume_from=resume_from,
        priority='HIGH')
]

runner = dict(type='EpochBasedRunner', max_epochs=300)

evaluation = dict(interval=1, metric='bbox')

checkpoint_config = dict(interval=5)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]

cudnn_benchmark = True
