import torch
import torch.nn as nn


class ContextBlock(nn.Module):

    def __init__(self, inplanes, ratio,
                 context_modeling_type='att',
                 fusion_type='add'):
        super(ContextBlock, self).__init__()
        assert context_modeling_type in ['att', 'avg']
        assert fusion_type in ['add', 'mul']

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.context_modeling_type = context_modeling_type
        self.fusion_type = fusion_type

        # context modeling
        if self.context_modeling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=1)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # transform
        self.transform = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
        )

        self.reset_parameters()


    def reset_parameters(self):
        if self.context_modeling_type == 'att':
            nn.init.kaiming_normal_(self.conv_mask.weight, mode='fan_in')
            nn.init.constant_(self.conv_mask.bias, 0)
            self.conv_mask.inited = True

        last_conv = self.transform[-1]
        nn.init.constant_(last_conv.weight, 0)
        nn.init.constant_(last_conv.bias, 0)

    def context_modeling(self, x):
        batch, channel, _, _ = x.size()
        if self.context_modeling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, -1)
            # [N, 1, H, W]
            context_att = self.conv_mask(x)
            # [N, H * W]
            context_att = self.softmax(context_att.view(batch, -1))
            # [N, H * W, 1]
            context_att = context_att.unsqueeze(2)
            # [N, C, 1, 1]
            context = torch.matmul(input_x, context_att).unsqueeze(3)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        context = self.context_modeling(x)
        out = x
        fusion_term = self.transform(context)
        if self.fusion_type == 'mul':
            out *= torch.sigmoid(fusion_term)
        else:
            out += fusion_term
        return out



