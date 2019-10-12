# -*- coding: utf-8 -*-
import torch
from torch import nn
import numpy as np


def pearson_coefficient(out, label):
    out_avg = np.mean(out)
    label_avg = np.mean(label)
    out_diff = out - out_avg
    label_diff = label - label_avg
    numerator = np.sum(out_diff*label_diff)
    denominator = np.sqrt(np.sum(pow(out_diff,2)))*np.sqrt(np.sum(pow(label_diff,2)))
    return numerator / denominator


# hyper-parameters
D_in, H1, H2, D_out = 444, 100, 20, 1


def neural_network(choice, init, data, device):
    base = nn.Sequential(nn.Linear(D_in, H1), nn.ReLU6(True),
                         nn.Linear(H1, H2), nn.ReLU6(True),
                         nn.Linear(H2, D_out)).to(device)
    if init == 'start':
        return base
    elif init == 'load':
        return torch.load('process/' + choice + '/' + data + '/checkpoint.tar').to(device)
