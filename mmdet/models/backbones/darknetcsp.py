import warnings

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.cnn.bricks.norm import build_norm_layer
from mmcv.runner import BaseModule
from mmcv.runner.fp16_utils import auto_fp16
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)


class Conv(ConvModule):
    # Standard convolution
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 groups=1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Mish'),
                 **kwargs):
        super(Conv, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2 if padding is None else padding,
            groups=groups,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)


class SimpleFocus(nn.Module):
    # Focus wh information into c-space
    def __init__(self,
                 in_channels,
                 out_channels,
                 b,
                 kernel_size=1,
                 stride=1,
                 groups=1,
                 init_cfg=None,
                 **kwargs):
        super(SimpleFocus, self).__init__()
        padding = kernel_size // 2
        self.b = b
        self.conv = Conv(in_channels, out_channels, kernel_size, stride, padding, groups, **kwargs)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        x = space_to_depth(x, self.b)
        return self.conv(x)


class Bottleneck(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 groups=1,
                 expansion=0.5,
                 init_cfg=None,
                 **kwargs):
        super(Bottleneck, self).__init__(init_cfg)
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = Conv(
            in_channels, hidden_channels, kernel_size=1, **kwargs)
        self.conv2 = Conv(
            hidden_channels,
            out_channels,
            kernel_size=3,
            groups=groups,
            **kwargs)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        if self.shortcut:
            return x + self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x))


class BottleneckCSP(BaseModule):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self,
                 in_channels,
                 out_channels,
                 repetition=1,
                 shortcut=True,
                 groups=1,
                 expansion=0.5,
                 csp_act_cfg=dict(type='Mish'),
                 init_cfg=None,
                 **kwargs):
        super(BottleneckCSP, self).__init__(init_cfg)
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = Conv(
            in_channels, hidden_channels, kernel_size=1, **kwargs)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(
            hidden_channels, hidden_channels, 1, 1, bias=False)
        self.conv4 = Conv(
            2 * hidden_channels, out_channels, kernel_size=1, **kwargs)
        csp_norm_cfg = kwargs.get('norm_cfg', dict(type='BN')).copy()
        self.bn = build_norm_layer(csp_norm_cfg, 2 * hidden_channels)[-1]
        csp_act_cfg_ = csp_act_cfg.copy()
        if csp_act_cfg_['type'] not in [
                'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish'
        ]:
            csp_act_cfg_.setdefault('inplace', True)
        self.csp_act = build_activation_layer(csp_act_cfg_)
        self.bottlenecks = nn.Sequential(*[
            Bottleneck(
                hidden_channels,
                hidden_channels,
                shortcut,
                groups,
                expansion=1.0,
                **kwargs) for _ in range(repetition)
        ])

    def forward(self, x):
        y1 = self.conv3(self.bottlenecks(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.csp_act(self.bn(torch.cat((y1, y2), dim=1))))


class BottleneckCSP2(BaseModule):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self,
                 in_channels,
                 out_channels,
                 repetition=1,
                 shortcut=False,
                 groups=1,
                 csp_act_cfg=dict(type='Mish'),
                 init_cfg=None,
                 **kwargs):
        super(BottleneckCSP2, self).__init__(init_cfg)
        hidden_channels = int(out_channels)  # hidden channels
        self.conv1 = Conv(
            in_channels, hidden_channels, kernel_size=1, **kwargs)
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, 1, 1, bias=False)
        self.conv3 = Conv(
            2 * hidden_channels, out_channels, kernel_size=1, **kwargs)
        csp_norm_cfg = kwargs.get('norm_cfg', dict(type='BN')).copy()
        self.bn = build_norm_layer(csp_norm_cfg, 2 * hidden_channels)[-1]
        csp_act_cfg_ = csp_act_cfg.copy()
        if csp_act_cfg_['type'] not in [
                'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish'
        ]:
            csp_act_cfg_.setdefault('inplace', True)
        self.csp_act = build_activation_layer(csp_act_cfg_)
        self.bottlenecks = nn.Sequential(*[
            Bottleneck(
                hidden_channels,
                hidden_channels,
                shortcut,
                groups,
                expansion=1.0,
                **kwargs) for _ in range(repetition)
        ])

    def forward(self, x):
        x1 = self.conv1(x)
        y1 = self.bottlenecks(x1)
        y2 = self.conv2(x1)
        return self.conv3(self.csp_act(self.bn(torch.cat((y1, y2), dim=1))))


class SPPV5(BaseModule):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self,
                 in_channels,
                 out_channels,
                 pooling_kernel_size=(5, 9, 13),
                 init_cfg=None,
                 **kwargs):
        super(SPPV5, self).__init__(init_cfg)
        hidden_channels = in_channels // 2  # hidden channels
        self.conv1 = Conv(
            in_channels, hidden_channels, kernel_size=1, **kwargs)
        self.conv2 = Conv(
            hidden_channels * (len(pooling_kernel_size) + 1),
            out_channels,
            kernel_size=1,
            **kwargs)
        self.maxpools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2)
            for x in pooling_kernel_size
        ])

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(
            torch.cat([x] + [maxpool(x) for maxpool in self.maxpools], 1))


class SPPV4(BaseModule):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=0.5,
                 pooling_kernel_size=(5, 9, 13),
                 csp_act_cfg=dict(type='Mish'),
                 init_cfg=None,
                 **kwargs):
        super(SPPV4, self).__init__(init_cfg)
        hidden_channels = int(2 * out_channels * expansion)  # hidden channels
        self.conv1 = Conv(
            in_channels, hidden_channels, kernel_size=1, **kwargs)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1, 1, bias=False)
        self.conv3 = Conv(
            hidden_channels, hidden_channels, kernel_size=3, **kwargs)
        self.conv4 = Conv(
            hidden_channels, hidden_channels, kernel_size=1, **kwargs)
        self.maxpools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2)
            for x in pooling_kernel_size
        ])
        self.conv5 = Conv(
            4 * hidden_channels, hidden_channels, kernel_size=1, **kwargs)
        self.conv6 = Conv(
            hidden_channels, hidden_channels, kernel_size=3, **kwargs)
        csp_norm_cfg = kwargs.get('norm_cfg', dict(type='BN')).copy()
        self.bn = build_norm_layer(csp_norm_cfg, 2 * hidden_channels)[-1]
        csp_act_cfg_ = csp_act_cfg.copy()
        if csp_act_cfg_['type'] not in [
                'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish'
        ]:
            csp_act_cfg_.setdefault('inplace', True)
        self.csp_act = build_activation_layer(csp_act_cfg_)
        self.conv7 = Conv(
            2 * hidden_channels, out_channels, kernel_size=1, **kwargs)

    def forward(self, x):
        x1 = self.conv4(self.conv3(self.conv1(x)))
        y1 = self.conv6(
            self.conv5(
                torch.cat([x1] + [maxpool(x1) for maxpool in self.maxpools],
                          1)))
        y2 = self.conv2(x)
        return self.conv7(self.csp_act(self.bn(torch.cat((y1, y2), dim=1))))


class Focus(BaseModule):
    # Focus wh information into c-space
    # Implement with ordinary Conv2d with
    # doubled kernel/padding size & stride 2
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 groups=1,
                 init_cfg=None,
                 **kwargs):
        super(Focus, self).__init__(init_cfg)
        padding = kernel_size // 2
        kernel_size *= 2
        padding *= 2
        stride *= 2
        self.conv = Conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            **kwargs)

    def forward(self, x):
        return self.conv(x)


class CSPStage(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 repetition,
                 init_cfg=None,
                 **kwargs):
        super(CSPStage, self).__init__(init_cfg)
        self.conv_downscale = Conv(
            in_channels, out_channels, kernel_size=3, stride=2, **kwargs)
        self.conv_csp = BottleneckCSP(out_channels, out_channels, repetition,
                                      **kwargs)

    def forward(self, x):
        return self.conv_csp(self.conv_downscale(x))


class SPPV5Stage(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 repetition,
                 init_cfg=None,
                 **kwargs):
        super(SPPV5Stage, self).__init__(init_cfg)
        self.conv_downscale = Conv(
            in_channels, out_channels, kernel_size=3, stride=2, **kwargs)
        self.spp = SPPV5(
            out_channels, out_channels, pooling_kernel_size=(5, 9, 13))
        # self.conv_csp = BottleneckCSP(out_channels, out_channels, repetition,
        #                               **kwargs)

    def forward(self, x):
        # return self.conv_csp(self.spp(self.conv_downscale(x)))
        return self.spp(self.conv_downscale(x))


class SPPV4Stage(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 repetition,
                 init_cfg=None,
                 **kwargs):
        super(SPPV4Stage, self).__init__(init_cfg)
        self.conv_downscale = Conv(
            in_channels, out_channels * 2, kernel_size=3, stride=2, **kwargs)
        self.conv_csp = BottleneckCSP(out_channels * 2, out_channels * 2,
                                      repetition, **kwargs)
        self.spp = SPPV4(
            out_channels * 2, out_channels, pooling_kernel_size=(5, 9, 13))

    def forward(self, x):
        return self.spp(self.conv_csp(self.conv_downscale(x)))


class BottleneckStage(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 repetition,
                 init_cfg=None,
                 **kwargs):
        super(BottleneckStage, self).__init__(init_cfg)
        self.conv_downscale = Conv(
            in_channels, out_channels, kernel_size=3, stride=2, **kwargs)
        self.conv_bottleneck = Bottleneck(out_channels, out_channels,
                                          repetition, **kwargs)

    def forward(self, x):
        return self.conv_bottleneck(self.conv_downscale(x))


@BACKBONES.register_module()
class DarknetCSP(BaseModule):
    """Darknet backbone.

    Args:
        scale (int): scale of DarknetCSP. 's'|'x'|'m'|'l'|
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Mish').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
    """

    arch_settings = {
        'v4s5p': [['conv', 'bottleneck', 'csp', 'csp', 'csp', 'sppv4'],
                  [None, 1, 1, 3, 3, 1], [16, 32, 64, 128, 256, 256]],
        'v4m5p': [['conv', 'bottleneck', 'csp', 'csp', 'csp', 'sppv4'],
                  [None, 1, 1, 5, 5, 3], [24, 48, 96, 192, 384, 384]],
        'v4l5p': [['conv', 'bottleneck', 'csp', 'csp', 'csp', 'sppv4'],
                  [None, 1, 2, 8, 8, 4], [32, 64, 128, 256, 512, 512]],
        'v4x5p': [['conv', 'bottleneck', 'csp', 'csp', 'csp', 'sppv4'],
                  [None, 1, 3, 11, 11, 5], [40, 80, 160, 320, 640, 640]],
        'v4l6p': [['conv', 'csp', 'csp', 'csp', 'csp', 'csp', 'sppv4'],
                  [None, 1, 3, 15, 15, 7, 7],
                  [32, 64, 128, 256, 512, 1024, 512]],
        'v4x7p': [['conv', 'csp', 'csp', 'csp', 'csp', 'csp', 'csp', 'sppv4'],
                  [None, 1, 3, 15, 15, 7, 7, 7],
                  [40, 80, 160, 320, 640, 1280, 1280, 640]],
        'v5s5p': [['focus', 'csp', 'csp', 'csp', 'sppv5'], [None, 1, 3, 3, 1],
                  [32, 64, 128, 256, 512]],
        's2dcsp': [['focus', 'csp', 'csp', 'csp', 'sppv5'], [None, 1, 1, 1, 1],
                  [32, 64, 128, 256, 512]],
        'v5m5p': [['focus', 'csp', 'csp', 'csp', 'sppv5'], [None, 2, 6, 6, 2],
                  [48, 96, 192, 384, 768]],
        'v5l5p': [['focus', 'csp', 'csp', 'csp', 'sppv5'], [None, 3, 9, 9, 3],
                  [64, 128, 256, 512, 1024]],
        'v5x5p': [['focus', 'csp', 'csp', 'csp', 'sppv5'],
                  [None, 4, 12, 12, 4], [80, 160, 320, 640, 1280]],
    }

    def __init__(self,
                 scale='x5p',
                 out_indices=(3, 4, 5),
                 frozen_stages=-1,
                 norm_cfg=dict(
                     type='BN', requires_grad=True, eps=0.001, momentum=0.03),
                 act_cfg=dict(type='Mish'),
                 csp_act_cfg=dict(type='Mish'),
                 norm_eval=False,
                 pretrained=None,
                 init_cfg=None):
        super(DarknetCSP, self).__init__(init_cfg)

        if isinstance(scale, str):
            if scale not in self.arch_settings:
                raise KeyError(f'invalid scale {scale} for DarknetCSP')
            stage, repetition, channels = self.arch_settings[scale]
        else:
            stage, repetition, channels = scale

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        cfg = dict(
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            csp_act_cfg=csp_act_cfg,
            init_cfg=init_cfg)

        self.layers = []
        cin = 3
        for i, (stg, rep, cout) in enumerate(zip(stage, repetition, channels)):
            layer_name = f'{stg}{i}'
            self.layers.append(layer_name)
            if stg == 'conv':
                self.add_module(layer_name, Conv(cin, cout, 3, **cfg))
            elif stg == 'bottleneck':
                self.add_module(layer_name,
                                BottleneckStage(cin, cout, rep, **cfg))
            elif stg == 'csp':
                self.add_module(layer_name, CSPStage(cin, cout, rep, **cfg))
            elif stg == 'focus':
                self.add_module(layer_name, Focus(cin, cout, 3, **cfg))
            elif stg == 'sppv4':
                self.add_module(layer_name, SPPV4Stage(cin, cout, rep, **cfg))
            elif stg == 'sppv5':
                self.add_module(layer_name, SPPV5Stage(cin, cout, rep, **cfg))
            else:
                raise NotImplementedError
            cin = cout

        self.norm_eval = norm_eval

        self.fp16_enabled = False

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

    @auto_fp16()
    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        # return tuple(outs)
        return outs

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)


    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(0, self.frozen_stages):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(DarknetCSP, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
