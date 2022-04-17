_base_ = '../yolov4/yolov4m_coco_mosaic.py'

model = dict(
    backbone=dict(scale='v5m5p', out_indices=[2, 3, 4]),
    neck=dict(type='YOLOV5Neck', in_channels=[192, 384, 768]),
)
