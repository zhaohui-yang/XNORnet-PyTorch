import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def countParams(state_dict):
    params_32 = 0
    params_1 = 0
    for key in state_dict.keys():
        if 'bin' in key:
            if 'full_precision' in key:
                continue
            if 'weight' in key:
                params_1 += state_dict[key].nelement()
                params_32 += state_dict[key].size(0)
            if 'bias' in key:
                params_32 += state_dict[key].nelement()
        else:
            params_32 += state_dict[key].nelement()
    return params_1, params_32

def countSize(state_dict):
    size = 0
    for key in state_dict.keys():
        if 'bin' in key:
            if 'full_precision' in key:
                continue
            if 'weight' in key:
                size += state_dict[key].nelement() / 8
                size += state_dict[key].size(0) * 4
            if 'bias' in key:
                size += state_dict[key].nelement() * 4
        else:
            size += state_dict[key].nelement() * 4
    return size
