import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer, ConvModule,
                      constant_init, kaiming_init)

from ..builder import BACKBONES

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# space to depth 
def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        # ------ convolution --------
        # norm conv
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)

        # ------ batchnormalization ---------------
        _, self.bn = build_norm_layer(dict(type='SyncBN', requires_grad=True), c2)

        # ------ activation ----------------------
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        kaiming_init(self.conv)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, b, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.b = b
        self.conv = Conv(c1, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        x = space_to_depth(x, self.b)
        return self.conv(x)


@BACKBONES.register_module()
class Space2DepthChain(nn.Module):
    def __init__(self, in_channels=16):
        super(Space2DepthChain, self).__init__()
        channels = [128, 256, 512, 1024, 2048]
        self.conv1 = Conv(3, 32, k=3, s=2, p=1, g=1, act=True)
        self.conv2 = Conv(32, 64, k=3, s=2, p=1, g=1, act=True)

        self.focus_8x = Focus(c1=64*4, c2=channels[0], b=2)
        self.focus_16x = Focus(c1=channels[0]*4, c2=channels[1], b=2)
        self.focus_32x = Focus(c1=channels[1]*4, c2=channels[2], b=2)
        self.focus_64x = Focus(c1=channels[2]*4, c2=channels[3], b=2)
        self.focus_128x = Focus(c1=channels[3]*4, c2=channels[4], b=2)
    
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(' model init ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return
    
    def forward(self, x):
        outs = []
        x_2x = self.conv1(x)
        x_4x = self.conv2(x_2x)
        x_8x = self.focus_8x(x_4x)
        outs.append(x_8x)
        x_16x = self.focus_16x(x_8x)
        outs.append(x_16x)
        x_32x = self.focus_32x(x_16x)
        outs.append(x_32x)
        x_64x = self.focus_64x(x_32x)
        outs.append(x_64x)
        x_128x = self.focus_128x(x_64x)
        outs.append(x_128x)
        
        return outs

