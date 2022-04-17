_base_ = '../yolov5/yolov5s_coco_mosaic.py'

model = dict(
    backbone=dict(
        norm_cfg=dict(
            type='SyncBN', requires_grad=True, eps=0.001, momentum=0.03)),
    neck=dict(
        norm_cfg=dict(
            type='SyncBN', requires_grad=True, eps=0.001, momentum=0.03)),
    bbox_head=dict(
        norm_cfg=dict(
            type='SyncBN', requires_grad=True, eps=0.001, momentum=0.03)),
)

optimizer = dict(lr=0.01)

data = dict(samples_per_gpu=16, workers_per_gpu=2)

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
fp16 = dict(
    loss_scale=dict(
        init_scale=2**16, mode='dynamic', scale_factor=2., scale_window=1000))
