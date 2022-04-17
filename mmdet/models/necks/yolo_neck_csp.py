import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.runner.fp16_utils import auto_fp16

from ..backbones.darknetcsp import BottleneckCSP, BottleneckCSP2, Conv
from ..builder import NECKS


@NECKS.register_module()
class YOLOV4Neck(BaseModule):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int or List[int]): Number of output channels (used at
            each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=None,
                 csp_repetition=3,
                 start_level=0,
                 end_level=-1,
                 norm_cfg=dict(
                     type='BN', requires_grad=True, eps=0.001, momentum=0.03),
                 act_cfg=dict(type='Mish'),
                 csp_act_cfg=dict(type='Mish'),
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=None):

        if init_cfg is None:
            init_cfg = [
                dict(type='Xavier', distribution='uniform', layer='Conv2d'),
                dict(
                    type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
            ]

        super(YOLOV4Neck, self).__init__(init_cfg)

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        if isinstance(out_channels, list):
            self.out_channels = out_channels
            num_outs = len(out_channels)
        else:
            assert num_outs is not None
            self.out_channels = [out_channels] * num_outs

        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs == self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        cfg = dict(
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            csp_act_cfg=csp_act_cfg,
            init_cfg=init_cfg)

        # 1x1 convs to shrink channels count before upsample and concat
        self.pre_upsample_convs = nn.ModuleList()

        # 1x1 convs to shrink backbone output channels count before concat
        self.backbone_pre_concat_convs = nn.ModuleList()

        # CSP convs to shrink channels after concat
        self.post_upsample_concat_csp = nn.ModuleList()

        # strided convs used to downsample
        self.downsample_convs = nn.ModuleList()

        # CSP convs after downsample
        self.post_downsample_concat_csp = nn.ModuleList()

        # yolov4 use 3x3 convs to process the final output
        self.out_convs = nn.ModuleList()

        # top-down path
        # from top level(smaller and deeper heat maps)
        # to bottom level(bigger and shallower heat maps) input index
        # starts with the topmost output of the backbone
        current_channels = in_channels[self.backbone_end_level - 1]
        to_bottom_up_concat_channels = []
        for i in range(self.backbone_end_level - 1, self.start_level, -1):
            bottom_channels = in_channels[i - 1]
            # yolov4 style
            target_channels = bottom_channels // 2

            # yolov4 send the input of this 1x1 conv to bottom up process flow
            # for concatenation
            to_bottom_up_concat_channels.append(current_channels)
            pre_up_conv = Conv(
                in_channels=current_channels,
                out_channels=target_channels,
                kernel_size=1,
                **cfg)

            backbone_pre_cat_conv = Conv(
                in_channels=bottom_channels,
                out_channels=target_channels,
                kernel_size=1,
                **cfg)

            post_upcat_csp = BottleneckCSP2(
                in_channels=2 * target_channels,
                # channel count doubles after concatenation
                out_channels=target_channels,
                repetition=csp_repetition,
                shortcut=False,
                **cfg)
            self.pre_upsample_convs.insert(0, pre_up_conv)
            self.backbone_pre_concat_convs.insert(0, backbone_pre_cat_conv)
            self.post_upsample_concat_csp.insert(0, post_upcat_csp)
            current_channels = target_channels

        # bottom-up path
        # from bottom level(bigger and shallower heat maps)
        # to top level(smaller and deeper heat maps)
        to_output_channels = [current_channels]
        for i in range(self.start_level, self.backbone_end_level - 1):
            top_channels = to_bottom_up_concat_channels.pop(-1)

            down_conv = Conv(
                in_channels=current_channels,
                out_channels=top_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                **cfg)

            post_downcat_csp = BottleneckCSP2(
                in_channels=2 * top_channels,
                # channel count doubles after concatenation
                out_channels=top_channels,
                repetition=csp_repetition,
                shortcut=False,
                **cfg)
            self.downsample_convs.append(down_conv)
            self.post_downsample_concat_csp.append(post_downcat_csp)
            to_output_channels.append(top_channels)
            current_channels = top_channels

        # build output conv
        for i in range(num_outs):
            before_conv_channels = to_output_channels[i]
            out_channels = self.out_channels[i]
            out_conv = Conv(
                in_channels=before_conv_channels,
                out_channels=out_channels,
                kernel_size=3,
                **cfg)
            self.out_convs.append(out_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        used_backbone_levels = self.backbone_end_level - self.start_level

        # build top-down path
        x = inputs[self.backbone_end_level - 1]
        bottom_up_merge = []

        for i in range(used_backbone_levels - 1, 0, -1):  # [2, 1]
            pre_up_conv = self.pre_upsample_convs[i - 1]
            backbone_pre_cat_conv = self.backbone_pre_concat_convs[i - 1]
            post_upcat_csp = self.post_upsample_concat_csp[i - 1]

            inputs_bottom = backbone_pre_cat_conv(inputs[self.start_level + i -
                                                         1])

            # yolov4 send the input of this 1x1 conv to bottom up process flow
            # for concatenation
            bottom_up_merge.append(x)
            x = pre_up_conv(x)

            if 'scale_factor' in self.upsample_cfg:
                x = F.interpolate(x, **self.upsample_cfg)
            else:
                bottom_shape = inputs_bottom.shape[2:]
                x = F.interpolate(x, size=bottom_shape, **self.upsample_cfg)

            x = torch.cat((inputs_bottom, x), dim=1)
            x = post_upcat_csp(x)

        # build additional bottom up path

        outs = [x]
        for i in range(self.backbone_end_level - self.start_level - 1):
            down_conv = self.downsample_convs[i]
            post_downcat_csp = self.post_downsample_concat_csp[i]
            x = down_conv(x)
            x = torch.cat((x, bottom_up_merge.pop(-1)), dim=1)
            x = post_downcat_csp(x)
            outs.append(x)

        # yolov4 use 3x3 convs to process the final output

        for i in range(len(outs)):
            outs[i] = self.out_convs[i](outs[i])

        return tuple(outs)

    def init_weights(self):
        """Initialize the weights of module."""
        # init is done in ConvModule
        pass


@NECKS.register_module()
class YOLOV5Neck(BaseModule):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int or List[int]): Number of output channels (used at
            each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=None,
                 csp_repetition=3,
                 start_level=0,
                 end_level=-1,
                 norm_cfg=dict(
                     type='BN', requires_grad=True, eps=0.001, momentum=0.03),
                 act_cfg=dict(type='Mish'),
                 csp_act_cfg=dict(type='Mish'),
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=None):

        if init_cfg is None:
            init_cfg = [
                dict(type='Xavier', distribution='uniform', layer='Conv2d'),
                dict(
                    type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
            ]

        super(YOLOV5Neck, self).__init__(init_cfg)

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        if isinstance(out_channels, list):
            self.out_channels = out_channels
            num_outs = len(out_channels)
        else:
            assert num_outs is not None
            self.out_channels = [out_channels] * num_outs

        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs == self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        cfg = dict(
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            csp_act_cfg=csp_act_cfg,
            init_cfg=init_cfg)

        # shrink channels count before upsample and concat
        self.pre_upsample_convs = nn.ModuleList()

        # yolov5 has no 1x1 conv before feeding the output of the backbone
        # to top down process flow for concatenation

        # CSP convs to shrink channels after concat
        self.post_upsample_concat_csp = nn.ModuleList()

        # convs for downsample
        self.downsample_convs = nn.ModuleList()

        # CSP convs after downsample
        self.post_downsample_concat_csp = nn.ModuleList()

        # yolov5 has no final 1x1 conv to process the final output

        # top-down path
        # from top level(smaller and deeper heat maps)
        # to bottom level(bigger and shallower heat maps) input index
        # starts with the topmost output of the backbone
        current_channels = in_channels[self.backbone_end_level - 1]
        to_bottom_up_concat_channels = []
        for i in range(self.backbone_end_level - 1, self.start_level, -1):
            bottom_channels = in_channels[i - 1]
            # yolov5 style
            target_channels = bottom_channels

            # yolov5 send the output of this 1x1 conv to bottom up process flow
            # for concatenation
            pre_up_conv = Conv(
                in_channels=current_channels,
                out_channels=target_channels,
                kernel_size=1,
                **cfg)
            to_bottom_up_concat_channels.append(target_channels)

            post_upcat_csp = BottleneckCSP(
                in_channels=2 * target_channels,
                # channel count doubles after concatenation
                out_channels=target_channels,
                repetition=csp_repetition,
                shortcut=False,
                **cfg)
            self.pre_upsample_convs.insert(0, pre_up_conv)
            self.post_upsample_concat_csp.insert(0, post_upcat_csp)

            current_channels = target_channels

        # bottom-up path
        # from bottom level(bigger and shallower heat maps)
        # to top level(smaller and deeper heat maps)
        to_output_channels = [current_channels]
        for i in range(self.start_level, self.backbone_end_level - 1):
            top_channels = to_bottom_up_concat_channels.pop(-1)
            # yolov5 style
            target_channels = self.out_channels[i - self.start_level + 1]

            down_conv = Conv(
                in_channels=current_channels,
                out_channels=top_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                **cfg)

            post_downcat_csp = BottleneckCSP(
                in_channels=2 * top_channels,
                out_channels=target_channels,
                repetition=csp_repetition,
                shortcut=False,
                **cfg)
            self.downsample_convs.append(down_conv)
            self.post_downsample_concat_csp.append(post_downcat_csp)
            to_output_channels.append(top_channels)
            current_channels = target_channels
        # yolov5 has no output 1x1 conv

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        used_backbone_levels = self.backbone_end_level - self.start_level

        # build top-down path
        x = inputs[self.backbone_end_level - 1]
        bottom_up_merge = []

        for i in range(used_backbone_levels - 1, 0, -1):  # [2, 1]
            pre_up_conv = self.pre_upsample_convs[i - 1]
            post_upcat_csp = self.post_upsample_concat_csp[i - 1]

            # yolov5 has no 1x1 conv before feeding the output of the backbone
            # to top down process flow for concatenation
            inputs_bottom = inputs[self.start_level + i - 1]

            # yolov5 send the output of this 1x1 conv to bottom up process flow
            # for concatenation
            x = pre_up_conv(x)
            bottom_up_merge.append(x)

            if 'scale_factor' in self.upsample_cfg:
                x = F.interpolate(x, **self.upsample_cfg)
            else:
                bottom_shape = inputs_bottom.shape[2:]
                x = F.interpolate(x, size=bottom_shape, **self.upsample_cfg)

            x = torch.cat((inputs_bottom, x), dim=1)
            x = post_upcat_csp(x)

        # build additional bottom up path

        outs = [x]
        for i in range(self.backbone_end_level - self.start_level - 1):
            down_conv = self.downsample_convs[i]
            post_downcat_csp = self.post_downsample_concat_csp[i]
            x = down_conv(x)
            x = torch.cat((x, bottom_up_merge.pop(-1)), dim=1)
            x = post_downcat_csp(x)
            outs.append(x)

        # yolov5 has no final 3x3 conv to process the final output

        return tuple(outs)

    def init_weights(self):
        """Initialize the weights of module."""
        # init is done in ConvModule
        pass
