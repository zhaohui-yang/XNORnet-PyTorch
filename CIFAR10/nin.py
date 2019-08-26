import torch
import torch.nn as nn

import init
from modules import *

class NIN(nn.Module):
    def __init__(self):
        super(NIN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 192, kernel_size = 5, stride = 1, padding = 2)
        self.bn1 = nn.BatchNorm2d(192, eps=1e-4, momentum = 0.1, affine = False)

        self.conv2_1 = BNConvReLU(192, 160, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = BNConvReLU(160, 96, kernel_size=1, stride=1, padding=0)

        self.conv3_1 = BNConvReLU(96, 192, kernel_size=5, stride=1, padding=2, dropout_ratio=0.5)
        self.conv3_2 = BNConvReLU(192, 192, kernel_size=1, stride=1, padding=0)
        self.conv3_3 = BNConvReLU(192, 192, kernel_size=1, stride=1, padding=0)

        self.conv4_1 = BNConvReLU(192, 192, kernel_size=3, stride=1, padding=1, dropout_ratio=0.5)
        self.conv4_2 = BNConvReLU(192, 192, kernel_size=1, stride=1, padding=0)

        self.bn5 = nn.BatchNorm2d(192, eps = 1e-4, momentum = 0.1, affine = False)
        self.conv5 = nn.Conv2d(192, 10, kernel_size = 1, stride = 1, padding = 0)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)
                    
    def forward(self, x, params, flops):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min = 0.01)
                    
        x = F.relu(self.bn1(self.conv1(x)))
        params = params + self.conv1.weight.nelement()
        flops = flops + self.conv1.weight.nelement() * 5 * 5 * x.size(-1) * x.size(-2)

        x, params, flops = self.conv2_1(x, params, flops)
        x, params, flops = self.conv2_2(x, params, flops)
        x = F.max_pool2d(x, kernel_size = 3, stride = 2, padding = 1)

        x, params, flops = self.conv3_1(x, params, flops)
        x, params, flops = self.conv3_2(x, params, flops)
        x, params, flops = self.conv3_3(x, params, flops)
        x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)

        x, params, flops = self.conv4_1(x, params, flops)
        x, params, flops = self.conv4_2(x, params, flops)

        x = F.relu(self.conv5(self.bn5(x)))
        params = params + self.conv5.weight.nelement()
        flops = flops + self.conv5.weight.nelement() * 1 * 1 * x.size(-1) * x.size(-2)
        x = F.avg_pool2d(x, kernel_size = 8)
        x = x.squeeze()
        return x, params, flops
