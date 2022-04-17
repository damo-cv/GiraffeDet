_base_ = '../yolov4/yolov4x_coco_mosaic.py'

model = dict(
    backbone=dict(scale='v5x5p', out_indices=[2, 3, 4]),
    neck=dict(type='YOLOV5Neck', in_channels=[320, 640, 1280]),
)
