_base_ = '../yolov4/yolov4s_coco_mosaic.py'

model = dict(
    backbone=dict(scale='v5s5p', out_indices=[2, 3, 4]),
    neck=dict(type='YOLOV5Neck', in_channels=[128, 256, 512]),
)
