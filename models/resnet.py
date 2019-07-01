import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from .context_block import ContextBlock


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, 
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, context_block_cfg=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 context_block_cfg=None):
        super(Bottleneck, self).__init__()
        assert context_block_cfg is None or isinstance(context_block_cfg, dict)

        self.with_context_block = context_block_cfg is not None

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_context_block:
            context_block_inplanes = planes * self.expansion
            self.context_block = ContextBlock(inplanes=context_block_inplanes, **context_block_cfg)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) 
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.with_context_block:
            out = self.context_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # block, layers, out_feat_dim, url
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2), 512, 'https://download.pytorch.org/models/resnet18-5c106cde.pth'),
        34: (BasicBlock, (3, 4, 6, 3), 512, 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'),
        50: (Bottleneck, (3, 4, 6, 3), 2048, 'https://download.pytorch.org/models/resnet50-19c8e357.pth'),
        101: (Bottleneck, (3, 4, 23, 3), 2048, 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'),
        152: (Bottleneck, (3, 8, 36, 3), 2048, 'https://download.pytorch.org/models/resnet152-b121ed2d.pth')
    }

    def __init__(self, depth,
                 stage_with_context_block=[False, False, False, False],
                 context_block_cfg=None,
                 pretrained=False):
        super(ResNet, self).__init__()
        assert depth in self.arch_settings
        block, layers, self.out_feat_dim, url = self.arch_settings[depth]

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 1,
            stage_with_context_block[0], context_block_cfg)
        self.layer2 = self._make_layer(block, 128, layers[1], 2,
            stage_with_context_block[1], context_block_cfg)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, 
            stage_with_context_block[2], context_block_cfg)
        self.layer4 = self._make_layer(block, 512, layers[3], 2,
            stage_with_context_block[3], context_block_cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if pretrained:
            self._load_pretrained(url)

    def _load_pretrained(self, url):
        print('loading from {}'.format(url))
        state_dict = load_state_dict_from_url(url)
        self.load_state_dict(state_dict, strict=False)
        print('load over!')

    def _make_layer(self, block, planes, blocks, stride=1,
                    with_context_block=False, context_block_cfg=None):
        if not with_context_block:
            context_block_cfg = None
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, context_block_cfg))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, context_block_cfg=context_block_cfg))

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








