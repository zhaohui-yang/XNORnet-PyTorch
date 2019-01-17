import torch
import torch.nn as nn

import init
from modules import *

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.bn1 = nn.BatchNorm2d(20, eps = 1e-4, momentum = 0.1, affine = False)
        self.bn2 = nn.BatchNorm2d(20, eps = 1e-4, momentum = 0.1, affine = True)
        self.bin_conv2 = XNORConv2d(20, 50, kernel_size = 5, stride = 1, padding = 0, groups = 1, dropout_ratio = 0)
        self.bn3 = nn.BatchNorm2d(50, eps = 1e-4, momentum = 0.1, affine = True)
        self.bin_fc1 = XNORLinear(50*4*4, 500, dropout_ratio = 0)
        self.fc2 = nn.Linear(500, 10)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)
                    
    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min = 0.01)
                    
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bin_conv2(self.bn2(x))), 2)
        x = self.bn3(x)
        x = x.view(-1, 50*4*4)
        x = F.relu(self.bin_fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(x, dim = 1)
        return x
