import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from typing import List, Tuple
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from functools import partial
from collections import OrderedDict

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2) 

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, bias=False, *args,
                 **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv, self.bias = expansion, downsampling, conv, bias
        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                                  stride=self.downsampling, bias=self.bias)
            })) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs)}))


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1

    def __init__(self, in_channels, out_channels, activation=nn.PReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=self.bias, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=self.bias),
        )


# ResNet.py (only the changed class shown)

class ResNetLayer(nn.Module):
    def __init__(
        self,
        Initial_channels,
        in_channels,
        out_channels,
        final_channels,
        block=ResNetBasicBlock,
        n=1,
        apply_sigmoid=False,          # <- NEW
        *args,
        **kwargs
    ):
        super().__init__()
        self.apply_sigmoid = apply_sigmoid   # <- NEW

        # downsampling = 2 if in_channels != out_channels else 1
        downsampling = 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

        self.conv_first = nn.Sequential(
            nn.Conv2d(Initial_channels, in_channels, kernel_size=3, padding=1),
            nn.PReLU(),
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, final_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.conv_first(x)
        x = self.blocks(x)
        x = self.conv_last(x)
        if self.apply_sigmoid:          # <- NEW
            x = torch.sigmoid(x)
        return x
