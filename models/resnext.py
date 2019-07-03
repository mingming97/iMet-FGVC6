import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from .resnet import Bottleneck as _Bottleneck
from .ops import conv1x1, conv3x3


class Bottleneck(_Bottleneck):

    def __init__(self, groups=1, base_width=4, *args, **kwargs):
        super(Bottleneck, self).__init__(*args, **kwargs)

        if groups == 1:
            width = self.planes
        else:
            width = int(self.planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(self.inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride=self.stride, groups=groups)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, self.planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(self.planes * self.expansion)


class ResNeXt(nn.Module):
    # block, layers, out_feat_dim, url
    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3), 2048, 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth'),
        101: (Bottleneck, (3, 4, 23, 3), 2048, 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth')
    }

    def __init__(self, depth, groups=1, base_width=4, 
                 stage_with_context_block=[False, False, False, False],
                 context_block_cfg=None,
                 pretrained=False):
        super(ResNeXt, self).__init__()
        assert depth in self.arch_settings
        block, layers, self.out_feat_dim, url = self.arch_settings[depth]
        if depth == 50:
            groups = 32
            base_width = 4
        else:
            groups = 32
            base_width = 8

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, groups, base_width,
            stage_with_context_block[0], context_block_cfg)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, groups, base_width,
            stage_with_context_block[1], context_block_cfg)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, groups, base_width,
            stage_with_context_block[2], context_block_cfg)
        self.layer4 = self._make_layer(block, 512, layers[3], 2, groups, base_width,
            stage_with_context_block[3], context_block_cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if pretrained:
            self._load_pretrained(url)

    def _load_pretrained(self, url):
        print('loading from {}'.format(url))
        state_dict = load_state_dict_from_url(url)
        self.load_state_dict(state_dict, strict=False)
        print('load over!')

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, base_width=4,
                    with_context_block=False, context_block_cfg=None):
        if not with_context_block:
            context_block_cfg = None
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(
            inplanes=self.inplanes, 
            planes=planes, 
            stride=stride, 
            downsample=downsample,
            groups=groups,
            base_width=base_width,
            context_block_cfg=context_block_cfg))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(
                inplanes=self.inplanes, 
                planes=planes,
                groups=groups,
                base_width=base_width,
                context_block_cfg=context_block_cfg))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x).view(x.size(0), -1)

        return x

