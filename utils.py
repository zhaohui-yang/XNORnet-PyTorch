import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def countParams(state_dict):
    params = 0
    for key in state_dict.keys():
        params += state_dict[key].nelement()
    return params

def countSize(state_dict):
    size = 0
    for key in state_dict.keys():
        if 'bin' in key:
            if 'full_precision' in key:
                continue
            if 'weight' in key:
                size += state_dict[key].nelement() / 8
            if 'bias' in key:
                size += state_dict[key].nelement() * 4
        else:
            size += state_dict[key].nelement() * 4
    return size
