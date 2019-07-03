import torch
import torch.nn as nn


class ContextBlock(nn.Module):

    def __init__(self, inplanes, ratio,
                 context_modeling_type='att',
                 fusion_type='add',
                 with_layernorm=True):
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
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # transform
        if with_layernorm:
            self.transform = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.transform = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
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
        batch, channel, height, width = x.size()
        if self.context_modeling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        context = self.context_modeling(x)
        out = x
        fusion_term = self.transform(context)
        if self.fusion_type == 'mul':
            out = out * torch.sigmoid(fusion_term)
        else:
            out = out + fusion_term
        return out



