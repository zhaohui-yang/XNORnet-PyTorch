from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinActive(torch.autograd.Function):
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        input = input.sign()
        return input
    
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class XNORConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, dropout_ratio):
        super(XNORConv2d, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.groups, self.dropout_ratio = in_channels, out_channels, kernel_size, stride, padding, groups, dropout_ratio
        if dropout_ratio !=0:
            self.dropout = nn.Dropout(dropout_ratio)
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, groups = groups)
        self.full_precision = nn.Parameter(torch.zeros(self.conv.weight.size()))
        self.full_precision.data.copy_(self.conv.weight.data)
        
    def forward(self, x):
        self.full_precision.data = self.full_precision.data - self.full_precision.data.mean(1, keepdim = True)
        self.full_precision.data.clamp_(-1, 1)

        x = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        mean_val = self.full_precision.abs().view(self.out_channels, -1).mean(1)
        self.mean_val = mean_val.view(-1)

        self.conv.weight.data.copy_(self.full_precision.data.sign() * self.mean_val.view(-1, 1, 1, 1))
        x = self.conv(x)
        return x

    def copy_grad(self):
        proxy = self.full_precision.abs().sign()
        proxy[self.full_precision.data.abs()>1] = 0
        binary_grad = self.conv.weight.grad * self.mean_val.view(-1, 1, 1, 1) * proxy

        mean_grad = self.conv.weight.data.sign() * self.conv.weight.grad
        mean_grad = mean_grad.view(self.out_channels, -1).mean(1).view(-1, 1, 1, 1)
        mean_grad = mean_grad * self.conv.weight.data.sign()

        self.full_precision.grad = binary_grad + mean_grad
        self.full_precision.grad = self.full_precision.grad * self.full_precision.data[0].nelement() * (1-1/self.full_precision.data.size(1))

class XNORLinear(nn.Module):
    def __init__(self, in_features, out_features, dropout_ratio):
        super(XNORLinear, self).__init__()
        self.in_features, self.out_features, self.dropout_ratio = in_features, out_features, dropout_ratio
        if dropout_ratio !=0:
            self.dropout = nn.Dropout(dropout_ratio)
        self.linear = nn.Linear(in_features = in_features, out_features = out_features)
        self.full_precision = nn.Parameter(torch.zeros(self.linear.weight.size()))
        self.full_precision.data.copy_(self.linear.weight.data)

    def forward(self, x):
        self.full_precision.data = self.full_precision.data - self.full_precision.data.mean(1, keepdim = True)
        self.full_precision.data.clamp_(-1, 1)

        x = BinActive()(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        mean_val = self.full_precision.abs().view(self.out_features, -1).mean(1)
        self.mean_val = mean_val

        self.linear.weight.data.copy_(self.full_precision.data.sign() * mean_val.view(-1, 1))
        x = self.linear(x)
        return x

    def copy_grad(self):
        proxy = self.full_precision.abs().sign()
        proxy[self.full_precision.data.abs()>1] = 0
        binary_grad = self.linear.weight.grad * self.mean_val.view(-1, 1) * proxy

        mean_grad = self.linear.weight.data.sign() * self.linear.weight.grad
        mean_grad = mean_grad.view(self.out_features, -1).mean(1).view(-1, 1)
        mean_grad = mean_grad * self.linear.weight.data.sign()

        self.full_precision.grad = binary_grad + mean_grad
        self.full_precision.grad = self.full_precision.grad * self.full_precision.data[0].nelement() * (1-1/self.full_precision.data.size(1))
        
    
