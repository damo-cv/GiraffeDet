cudnn_benchmark = True
# model settings
norm_cfg = dict(type='SyncBN', momentum=0.01, eps=1e-3, requires_grad=True)  # using SyncBN during training
# norm_cfg = dict(type='BN', momentum=0.01, eps=1e-3, requires_grad=True)  # using SyncBN during training
model = dict(
    type='RetinaNet',
    # pretrained='work_dirs/efficientdet_d1_rwight_yewu_640/latest.pth',
    # url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_d0-f3276ba8.pth',
    backbone=dict(
        type='EfficientNet',
        backbone_name='efficientnet_b1',
        backbone_indices=[2,3,4],
        pretrained_backbone=True,
        backbone_args=dict(drop_path_rate=0.2),
        alternate_init=False),
    neck=dict(
        type='BiFPN',
        min_level=3,
        max_level=7,
        num_levels=5,
        norm_layer=None,
        norm_kwargs=dict(eps=.001, momentum=.01),
        act_type='swish',
        fpn_config=None,
        fpn_name=None,
        fpn_channels=96,
        pad_type='',
        downsample_type='max',
        upsample_type='nearest',
        apply_resample_bn=True,
        conv_after_downsample=False,
        redundant_bias=False,
        fpn_cell_repeats=11,
        separable_conv=True,
        conv_bn_relu_pattern=False,
        feature_info=[dict(num_chs=40, reduction=8), dict(num_chs=112, reduction=16), dict(num_chs=320, reduction=32)],
        alternate_init=False),
    bbox_head=dict(
        type='GFocalHead',
        num_classes=6,
        in_channels=96,
        stacked_convs=4,
        feat_channels=96,
        norm_cfg = dict(type='SyncBN', momentum=0.01, eps=1e-3, requires_grad=True),
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
        loss_bbox=dict(type='IoULoss', loss_weight=2.0)))

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
    nms=dict(type='nms', iou_thr=0.4),
    max_per_img=100)
# dataset settings
dataset_type = 'YWDataset'
data_root = 'data/yewu/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
h_size = 384
w_size = 768
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(h_size, w_size),
        ratio_range=(0.2, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(h_size, w_size)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=(h_size, w_size)),
    dict(type='Pad', size_divisor=128),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
h_size = 384
w_size = 768
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(h_size, w_size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size=(h_size, w_size)),
            dict(type='Pad', size_divisor=128),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            # pretrain dataset
            ann_file=[
                data_root + 'voc_format_datasets/ImageSets/Main/object365_train_ppmn.txt',
                data_root + 'voc_format_datasets/ImageSets/Main/coco_train2017_person_motor_nonmotor.txt',
                data_root + 'voc_format_datasets/ImageSets/Main/data4Augment.txt',
                data_root + 'voc_format_datasets/ImageSets/Main/train_all_yjwo.txt',
                data_root + 'voc_format_datasets/ImageSets/Main/data_group_A.txt',
                data_root + 'voc_format_datasets/ImageSets/Main/train_hp_batch1.txt',
                data_root + 'voc_format_datasets/ImageSets/Main/train_hp_batch2.txt',
                data_root + 'voc_format_datasets/ImageSets/Main/train_hp_batch3.txt'
            ],
            img_prefix=[data_root + 'voc_format_datasets/', data_root + 'voc_format_datasets/',
                        data_root + 'voc_format_datasets/', data_root + 'voc_format_datasets/',
                        data_root + 'voc_format_datasets/', data_root + 'voc_format_datasets/',
                        data_root + 'voc_format_datasets/', data_root + 'voc_format_datasets/'
                       ],
            # finetune dataset
            # ann_file=[
            #     data_root + 'voc_format_datasets/ImageSets/Main/train_all_vcs.txt',
            #     data_root + 'voc_format_datasets/ImageSets/Main/data_group_A.txt',
            #     data_root + 'voc_format_datasets/ImageSets/Main/train_hp_batch1.txt',
            #     data_root + 'voc_format_datasets/ImageSets/Main/train_hp_batch2.txt',
            #     data_root + 'voc_format_datasets/ImageSets/Main/train_hp_batch3.txt',
            #     data_root + 'voc_format_datasets/ImageSets/Main/hgy_face.txt',
            # ],
            # img_prefix=[data_root + 'voc_format_datasets/', data_root + 'voc_format_datasets/',
            #             data_root + 'voc_format_datasets/', data_root + 'voc_format_datasets/',
            #             data_root + 'voc_format_datasets/', data_root + 'voc_format_datasets/'
            #            ],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'voc_format_datasets/ImageSets/Main/val_hp_batch1.txt',
        # ann_file=data_root + 'voc_format_datasets/ImageSets/Main/hp_220200810_val_5000.txt',
        img_prefix=data_root + 'voc_format_datasets/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'voc_format_datasets/ImageSets/Main/val_hp_batch1.txt',
        # ann_file=data_root + 'voc_format_datasets/ImageSets/Main/hp_220200810_val_5000.txt',
        img_prefix=data_root + 'voc_format_datasets/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.04,
    momentum=0.9,
    weight_decay=4e-5)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
lr_config = dict(
      policy='step',
      step=[125, 143])
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=7500,
#     warmup_ratio=0.008,
#     min_lr=0.0001)
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 150
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/efficientdet_d1_rwight_yewu_multiscale'
# load_from='work_dirs/efficientdet_d1_rwight_yewu_640/latest.pth'
load_from = None
resume_from = None
workflow = [('train', 1)]
