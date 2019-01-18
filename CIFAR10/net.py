import torch
import torch.nn as nn

import init
from modules import *

class NIN(nn.Module):
    def __init__(self):
        super(NIN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 192, kernel_size = 5, stride = 1, padding = 2)
        self.bn1 = nn.BatchNorm2d(192, eps=1e-4, momentum = 0.1, affine = False)

        self.bn2_1 = nn.BatchNorm2d(192, eps = 1e-4, momentum = 0.1, affine = True)
        self.bin_conv2_1 = XNORConv2d(192, 160, kernel_size = 1, stride = 1, padding = 0, groups = 1, dropout_ratio = 0)
        self.bn2_2 = nn.BatchNorm2d(160, eps = 1e-4, momentum = 0.1, affine = True)
        self.bin_conv2_2 = XNORConv2d(160, 96, kernel_size = 1, stride = 1, padding = 0, groups = 1, dropout_ratio = 0)

        self.bn3_1 = nn.BatchNorm2d(96, eps = 1e-4, momentum = 0.1, affine = True)
        self.bin_conv3_1 = XNORConv2d(96, 192, kernel_size = 5, stride = 1, padding = 2, groups = 1, dropout_ratio = 0)
        self.bn3_2 = nn.BatchNorm2d(192, eps = 1e-4, momentum = 0.1, affine = True)
        self.bin_conv3_2 = XNORConv2d(192, 192, kernel_size = 1, stride = 1, padding = 0, groups = 1, dropout_ratio = 0)
        self.bn3_3 = nn.BatchNorm2d(192, eps = 1e-4, momentum = 0.1, affine = True)
        self.bin_conv3_3 = XNORConv2d(192, 192, kernel_size = 1, stride = 1, padding = 0, groups = 1, dropout_ratio = 0)

        self.bn4_1 = nn.BatchNorm2d(192, eps = 1e-4, momentum = 0.1, affine = True)
        self.bin_conv4_1 = XNORConv2d(192, 192, kernel_size = 3, stride = 1, padding = 1, groups = 1, dropout_ratio = 0)
        self.bn4_2 = nn.BatchNorm2d(192, eps = 1e-4, momentum = 0.1, affine = True)
        self.bin_conv4_2 = XNORConv2d(192, 192, kernel_size = 1, stride = 1, padding = 0, groups = 1, dropout_ratio = 0)

        self.bn5 = nn.BatchNorm2d(192, eps = 1e-4, momentum = 0.1, affine = False)
        self.conv5 = nn.Conv2d(192, 10, kernel_size = 1, stride = 1, padding = 0)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)
                    
    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min = 0.01)
                    
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bin_conv2_1(self.bn2_1(x)))
        x = F.relu(self.bin_conv2_2(self.bn2_2(x)))
        x = F.max_pool2d(x, kernel_size = 3, stride = 2, padding = 1)

        x = F.relu(self.bin_conv3_1(self.bn3_1(x)))
        x = F.relu(self.bin_conv3_2(self.bn3_2(x)))
        x = F.relu(self.bin_conv3_3(self.bn3_3(x)))
        x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = F.relu(self.bin_conv4_1(self.bn4_1(x)))
        x = F.relu(self.bin_conv4_2(self.bn4_2(x)))
        x = F.relu(self.conv5(self.bn5(x)))
        x = F.avg_pool2d(x, kernel_size = 8)
        x = x.squeeze()
        return x
        return x
