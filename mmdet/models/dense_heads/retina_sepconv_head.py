import numpy as np
import torch.nn as nn
from mmcv.cnn import kaiming_init

from ..builder import HEADS
from mmcv.cnn import bias_init_with_prob
from ..utils import SeparableConv2d
from .anchor_head import AnchorHead


@HEADS.register_module
class RetinaSepConvHead(AnchorHead):
    """"RetinaHead with separate BN and separable conv.

    In RetinaHead, conv/norm layers are shared across different FPN levels,
    while in RetinaSepBNHead, conv layers are shared across different FPN
    levels, but BN layers are separated.

    In EfficientDet, using separable conv as conv module.
    """

    def __init__(self,
                 num_classes,
                 num_ins,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_ins = num_ins
        octave_scales = np.array(
            [2 ** (i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale

        anchor_generator=dict(
            type='AnchorGenerator',
            scales=anchor_scales,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128])
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=(.0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0))
        super(RetinaSepConvHead, self).__init__(
            num_classes, in_channels, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.num_ins):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(
                    SeparableConv2d(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        activation="Swish",
                        bias=True,
                        norm_cfg=self.norm_cfg))
                reg_convs.append(
                    SeparableConv2d(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        activation="Swish",
                        bias=True,
                        norm_cfg=self.norm_cfg))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
        for i in range(self.stacked_convs):
            for j in range(1, self.num_ins):
                self.cls_convs[j][i].depthwise = self.cls_convs[0][i].depthwise
                self.cls_convs[j][i].pointwise.conv = self.cls_convs[0][i].pointwise.conv
                self.reg_convs[j][i].depthwise = self.reg_convs[0][i].depthwise
                self.reg_convs[j][i].pointwise.conv = self.reg_convs[0][i].pointwise.conv
        self.retina_cls = SeparableConv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1,
            bias=True,
            norm_cfg=None)
        self.retina_reg = SeparableConv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1, bias=True, norm_cfg=None)

    def init_weights(self):
        for m in self.cls_convs[0]:
            kaiming_init(m.depthwise, mode='fan_in')
            kaiming_init(m.pointwise.conv, mode='fan_in')
        for m in self.reg_convs[0]:
            kaiming_init(m.depthwise, mode='fan_in')
            kaiming_init(m.pointwise.conv, mode='fan_in')
        bias_cls = bias_init_with_prob(0.01)
        kaiming_init(self.retina_cls.depthwise, mode='fan_in')
        kaiming_init(self.retina_cls.pointwise.conv, mode='fan_in', bias=bias_cls)
        kaiming_init(self.retina_reg.depthwise, mode='fan_in')
        kaiming_init(self.retina_reg.pointwise.conv, mode='fan_in')

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        for i, x in enumerate(feats):
            cls_feat = feats[i]
            reg_feat = feats[i]
            for cls_conv in self.cls_convs[i]:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs[i]:
                reg_feat = reg_conv(reg_feat)
            cls_score = self.retina_cls(cls_feat)
            bbox_pred = self.retina_reg(reg_feat)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return cls_scores, bbox_preds
