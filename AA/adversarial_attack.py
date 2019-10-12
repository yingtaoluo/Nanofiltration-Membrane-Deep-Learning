# -*- coding: utf-8 -*-
import argparse
import torch
from torch.autograd import Variable
from torch import nn
from model import pearson_coefficient
import numpy as np
torch.manual_seed(2)


def to_numpy(x):
    return x.cpu().data.numpy() if device == 'cuda' else x.detach().numpy()


device = 'cuda'
batch_size, epoch = 200, 3
learning_rate, regularization = 1e-3, 3e-3
criterion = nn.MSELoss(reduction='mean')
epsilon = 0.1  # adjust attack strength at here

model = torch.load('process/rejection/1/checkpoint.tar').to(device)
dps = 'H:/interfacial_data/data/1'
train_numpy_data = np.load(dps + "/train_data.npy")  # size (320000,446)
test_numpy_data = np.load(dps + "/test_data.npy")  # size (29290, 446)
valid_numpy_data = np.load(dps + "/valid_data.npy")  # size (70700, 446)

train_input = torch.FloatTensor(train_numpy_data[:, 0:444]).to(device)
print(train_input)
train_label = torch.FloatTensor(train_numpy_data[:, 445:446]).to(device)  # rej
train_input = Variable(train_input, requires_grad=True)

test_input = torch.FloatTensor(test_numpy_data[:, 0:444]).to(device)
test_label = torch.FloatTensor(test_numpy_data[:, 445:446]).to(device)  # rej

valid_input = torch.FloatTensor(valid_numpy_data[:, 0:444]).to(device)
valid_label = torch.FloatTensor(valid_numpy_data[:, 445:446]).to(device)  # rej

train_predict = model(train_input)
loss_train = criterion(train_predict, train_label)

loss_train.backward()
print(train_input.grad)
adv_input = train_input + epsilon * torch.sign(train_input.grad)
adv_predict = model(adv_input)
loss_adv = criterion(adv_predict, train_label)

print("train_loss:{}, adv_loss:{}".format(loss_train, loss_adv))

test_predict = model(test_input)
loss_test = criterion(test_predict, test_label)
valid_predict = model(valid_input)
loss_valid = criterion(valid_predict, valid_label)

print("test_loss:{}, valid_loss:{}".format(loss_test, loss_valid))

train_out = train_predict - train_label
train_error = np.abs(to_numpy(train_out)).mean()
adv_out = adv_predict - train_label
adv_error = np.abs(to_numpy(adv_out)).mean()
test_out = test_predict - test_label
test_error = np.abs(to_numpy(test_out)).mean()
valid_out = valid_predict - valid_label
valid_error = np.abs(to_numpy(valid_out)).mean()

print('train_error: {:.4}, adv_error: {:.4}, test_error: {:.4}, '
      'valid_error: {:.4}'.format(train_error, adv_error, test_error, valid_error))

p_train = pearson_coefficient(to_numpy(train_predict), to_numpy(train_label))
print('training pearson coefficient: {:.4}'.format(p_train))

p_adv = pearson_coefficient(to_numpy(adv_predict), to_numpy(train_label))
print('adversarial pearson coefficient: {:.4}'.format(p_adv))

p_test = pearson_coefficient(to_numpy(test_predict), to_numpy(test_label))
print('testing pearson coefficient: {:.4}'.format(p_test))

p_valid = pearson_coefficient(to_numpy(valid_predict), to_numpy(valid_label))
print('validation pearson coefficient: {:.4}'.format(p_valid))






