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


D_in, H1, H2, D_out = 148, 100, 20, 1


def neural_network(choice, init, data, device):
    D_in = 160 if data[0:2] == '78' else 148
    if init == 'start':
        return nn.Sequential(nn.Linear(D_in, H1), nn.ReLU(True),
                             nn.Linear(H1, H2), nn.ReLU(True),
                             nn.Linear(H2, D_out)).to(device)
    elif init == 'load':
        return torch.load('process/' + choice + '/' + data + '/checkpoint.tar').to(device)
    else:
        print("Not Implemented! Error!")
