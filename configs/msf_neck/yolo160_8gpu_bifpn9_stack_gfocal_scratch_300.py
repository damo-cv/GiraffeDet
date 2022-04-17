cudnn_benchmark = True
_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='SingleStageDetector',
    # pretrained='torchvision://resnet50',
    pretrained=None,
    backbone=dict(
        type='DarknetCSP',
        scale='v5s5p',
        out_indices=(2, 3, 4),
    ),
    neck=dict(
        type='BiFPN',
        min_level=3,
        max_level=7,
        num_levels=5,
        norm_layer=None,
        norm_kwargs=dict(eps=.001, momentum=.01),
        act_type='relu',
        fpn_config=None,
        fpn_name=None,
        fpn_channels=[128, 160, 192, 224, 256],
        out_fpn_channels=[192, 192, 192, 192, 192],
        pad_type='',
        downsample_type='max',
        upsample_type='nearest',
        apply_resample_bn=True,
        conv_after_downsample=False,
        redundant_bias=False,
        fpn_cell_repeats=8,
        separable_conv=False,
        conv_bn_relu_pattern=False,
        feature_info=[dict(num_chs=128, reduction=8), dict(num_chs=256, reduction=16), dict(num_chs=512, reduction=32)],
        alternate_init=False),
    bbox_head=dict(
        type='GFocalHead',
        num_classes=80,
        in_channels=192,
        stacked_convs=4,
        feat_channels=192,
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
    # assigner=dict(
    #     type='MaxIoUAssigner',
    #     pos_iou_thr=0.5,
    #     neg_iou_thr=0.5,
    #     min_pos_iou=0,
    #     ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(
#    type='SGD',
#    lr=0.01,
#    momentum=0.9,
#    weight_decay=0.0001,
#    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))


# learning policy
lr_config = dict(warmup_ratio=0.1, step=[267, 291])
total_epochs = 300
# multi-scale training
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
#     #dict(
#     #    type='Resize',
#     #    img_scale=[(1333, 480), (1333, 960)],
#     #    multiscale_mode='range',
#     #    keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=128),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
img_norm_cfg = dict(mean=[114, 114, 114], std=[255, 255, 255], to_rgb=True)
train_pipeline = [
    dict(
        type='MosaicPipeline',
        individual_pipeline=[
            # dict(type='LoadImageFromFile', im_decode_backend='turbojpeg'),
            dict(type='LoadImageFromFile'),
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
data = dict(train=dict(pipeline=train_pipeline))


dataset_type = 'CocoDataset'
data_root = 'data/coco/'
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(640, 640),
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
    samples_per_gpu=4,
    workers_per_gpu=4,
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

