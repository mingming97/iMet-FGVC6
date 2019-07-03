import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from collections import OrderedDict
import re

from .ops import conv1x1, conv3x3
from .context_block import ContextBlock


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, 
                 context_block_cfg=None):
        assert context_block_cfg is None or isinstance(context_block_cfg, dict)

        super(_DenseLayer, self).__init__()
        if context_block_cfg is not None:
            self.add_module('context_block', ContextBlock(inplanes=num_input_features, **context_block_cfg))

        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', conv1x1(num_input_features, bn_size * growth_rate))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', conv3x3(bn_size * growth_rate, growth_rate))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, 
                 context_block_cfg=None):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, context_block_cfg)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', conv1x1(num_input_features, num_output_features))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):

    # num_init_features, growth_rate, block_config, url
    arch_settings = {
        121: (64, 32, (6, 12, 24, 16), 1024, 'https://download.pytorch.org/models/densenet121-a639ec97.pth'),
        169: (64, 32, (6, 12, 32, 32), 1664, 'https://download.pytorch.org/models/densenet169-b2777c0a.pth'),
        201: (64, 32, (6, 12, 48, 32), 1920, 'https://download.pytorch.org/models/densenet201-c1103571.pth'),
        161: (96, 48, (6, 12, 36, 24), 2208, 'https://download.pytorch.org/models/densenet161-8d451a50.pth')
    }
    def __init__(self, depth, bn_size=4, drop_rate=0, context_block_cfg=None, pretrained=False):
        super(DenseNet, self).__init__()
        assert depth in self.arch_settings
        num_init_features, growth_rate, block_config, self.out_feat_dim, url = self.arch_settings[depth]

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.with_context_block = context_block_cfg is not None

        if self.with_context_block:
            self.features.add_module('context_block_0a', ContextBlock(inplanes=num_init_features, **context_block_cfg))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if self.with_context_block:
                self.features.add_module('context_block_%da' % (i + 1), ContextBlock(inplanes=num_features, **context_block_cfg))

            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                context_block_cfg=context_block_cfg)
            self.features.add_module('denseblock%d' % (i + 1), block)

            num_features += num_layers * growth_rate

            if i != len(block_config) - 1:
                if self.with_context_block:
                    self.features.add_module('context_block_%db' % (i + 1), ContextBlock(inplanes=num_features, **context_block_cfg))

                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        if self.with_context_block:
            self.features.add_module('context_block_0b', ContextBlock(inplanes=num_features, **context_block_cfg))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if pretrained:
            self._load_pretrained(url)

    def _load_pretrained(self, url):
        print('loading from {}'.format(url))
        state_dict = load_state_dict_from_url(url)
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        self.load_state_dict(state_dict, strict=False)
        print('load over')

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.avg_pool(features).view(features.size(0), -1)
        return out