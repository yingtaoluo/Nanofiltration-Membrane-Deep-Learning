# -*- coding: utf-8 -*-
import os
import argparse
import torch
from torch import nn, optim
import torch.utils.data as data
import numpy as np
from model import pearson_coefficient, neural_network
torch.manual_seed(2)


def item(x):
    return x.cpu().data.item() if args.device == 'cuda' else x.item()


def to_numpy(x):
    return x.cpu().data.numpy() if args.device == 'cuda' else x.detach().numpy()


def weight_init(m):
    if args.choice == 'rejection':
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            nn.init.constant_(m.bias, 0)
    elif args.choice == 'permeability':
        if isinstance(m, nn.Linear):
            # nn.init.xavier_normal_(m.weight, mean=0, std=1)
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)


'''command prompt resolution'''
parser = argparse.ArgumentParser()
parser.add_argument('choice', type=str, choices=['rejection', 'permeability'],
                    help='choose to predict rejection or permeability')
parser.add_argument('init', type=str, choices=['load', 'start'],
                    help='choose to load the pre-trained parameters or randomly initialize parameters')
parser.add_argument('data', type=str, choices=['72_0', '72_1', '72_2', '78_0', '78_1', '78_2'],
                    help='choose which dataset to load')
# other auxiliary factors
aux_args = parser.add_argument_group('auxiliary')
aux_args.add_argument('--optim', type=str, choices=['adam', 'sgd'],
                      help='choose to use adam or sgd optimization algorithm')
aux_args.add_argument('--manual_init', type=bool, choices=[True, False],
                      help='choose to manually initialize parameters or use default initialization')
aux_args.add_argument('--save', type=bool, choices=[True, False],
                      help='choose to save the checkpoint or not')
aux_args.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                      help='choose to use cpu or gpu for computation')

parser.set_defaults(optim='adam',
                    manual_init=False,
                    save=True,
                    device='cuda')
args = parser.parse_args()


'''hyper-parameters'''
# These parameters are well selected after lots of experiments.
if args.choice == 'rejection':
    batch_size, epoch, lr = 200, 5, 1e-3
    regularization = 1e-2 if args.data[0:2] == '72' else 1e-1
    # manually pick the threshold to save the model parameters
    threshold = 0
elif args.choice == 'permeability':
    batch_size, epoch, lr = 200, 5, 5e-5
    # it is very likely to encounter overfitting when training permeability,
    # you can set regularization to a recommending range from 3e-1 to 3, .
    regularization = 0 if args.data[0:2] == '72' else 0
    # manually pick the threshold to save the model parameters
    threshold = -1


'''model structure'''
model = neural_network(choice=args.choice, init=args.init, data=args.data, device=args.device)
if args.init == 'start' and args.manual_init:
    model.apply(weight_init)
if args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=regularization)
elif args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=regularization, momentum=0.8)
else:
    raise NotImplementedError("Please select between adam and sgd")
# learning_rate decay
if args.choice == 'rejection':
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)
elif args.choice == 'permeability':
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)

criterion = nn.MSELoss(reduction='mean')


'''recording training processes'''
train_pearson_pro = []
test_pearson_pro = []
valid_pearson_pro = []
test_pro = []
train_pro = []
valid_pro = []
flag = 0


'''managing directory'''
# make process directory
if not os.path.isdir('./process'):
    os.makedirs('./process')
pps = './process' + '/' + args.choice + '/' + args.data
if not os.path.isdir(pps):
    os.makedirs(pps)
# make args dicts
with open(os.path.join(pps, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v)
                       for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
# data directory
dps = './data/' + args.data


'''loading data'''
train_numpy_data = np.load(dps + "/train_dataset.npy")
test_numpy_data = np.load(dps + "/test_dataset.npy")
valid_numpy_data = np.load(dps + "/valid_dataset.npy")
print('the size of train data: {}'.format(np.shape(train_numpy_data)))
print('the size of test data: {}'.format(np.shape(test_numpy_data)))
print('the size of valid data: {}'.format(np.shape(valid_numpy_data)))

# preparing data for the model
a = 148 if args.data[0:2] == '72' else 160
train_input = torch.FloatTensor(train_numpy_data[:, 0:a]).to(args.device)
if args.choice == 'rejection':
    train_label = torch.FloatTensor(train_numpy_data[:, a+1:a+2]).to(args.device)
elif args.choice == 'permeability':
    train_label = torch.FloatTensor(train_numpy_data[:, a:a+1]).to(args.device)
else:
    raise NotImplementedError()
train_data = data.TensorDataset(train_input, train_label)
train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_input = torch.FloatTensor(test_numpy_data[:, 0:a]).to(args.device)
if args.choice == 'rejection':
    test_label = torch.FloatTensor(test_numpy_data[:, a+1:a+2]).to(args.device)
elif args.choice == 'permeability':
    test_label = torch.FloatTensor(test_numpy_data[:, a:a+1]).to(args.device)
else:
    raise NotImplementedError()

valid_input = torch.FloatTensor(valid_numpy_data[:, 0:a]).to(args.device)
if args.choice == 'rejection':
    valid_label = torch.FloatTensor(valid_numpy_data[:, a+1:a+2]).to(args.device)
elif args.choice == 'permeability':
    valid_label = torch.FloatTensor(valid_numpy_data[:, a:a+1]).to(args.device)
else:
    raise NotImplementedError()


'''compute and record loss before training'''
train_predict = model(train_input)
loss_train = criterion(train_predict, train_label)
print('iteration: {}, train_loss: {:.4}'.format(0, item(loss_train)))
train_pro.append(item(loss_train))

test_predict = model(test_input)
loss_test = criterion(test_predict, test_label)
print('iteration: {}, test_loss: {:.4}'.format(0, item(loss_test)))
test_pro.append(item(loss_test))

valid_predict = model(valid_input)
loss_valid = criterion(valid_predict, valid_label)
print('iteration: {}, valid_loss: {:.4}'.format(0, item(loss_valid)))
valid_pro.append(item(loss_valid))

p_train = pearson_coefficient(to_numpy(train_predict), to_numpy(train_label))
print('iteration: {}, training pearson correlation coefficient: {}'.format(0, p_train))
train_pearson_pro.append(p_train)

p_test = pearson_coefficient(to_numpy(test_predict), to_numpy(test_label))
print('iteration: {}, testing pearson correlation coefficient: {}'.format(0, p_test))
test_pearson_pro.append(p_test)

p_valid = pearson_coefficient(to_numpy(valid_predict), to_numpy(valid_label))
print('iteration: {}, validation pearson correlation coefficient: {}'.format(0, p_valid))
valid_pearson_pro.append(p_valid)


'''training'''
for t in range(epoch):
    for step, data in enumerate(train_loader):
        # learning_rate decay
        scheduler.step()

        # train
        train_input, train_label = data
        train_predict = model(train_input)
        loss_train = criterion(train_predict, train_label)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        flag += 1

        print('iteration: {}, train_loss: {:.4}'.format(flag, item(loss_train)))
        train_pro.append(item(loss_train))

        # test
        test_predict = model(test_input)
        loss_test = criterion(test_predict, test_label)
        print('iteration: {}, test_loss: {:.4}'.format(flag, item(loss_test)))
        test_pro.append(item(loss_test))

        # valid
        valid_predict = model(valid_input)
        loss_valid = criterion(valid_predict, valid_label)
        print('iteration: {}, valid_loss: {:.4}'.format(flag, item(loss_valid)))
        valid_pro.append(item(loss_valid))

        # pearson
        p_train = pearson_coefficient(to_numpy(train_predict), to_numpy(train_label))
        print('iteration: {}, training pearson coefficient: {:.4}'.format(flag, p_train))
        train_pearson_pro.append(p_train)

        p_test = pearson_coefficient(to_numpy(test_predict), to_numpy(test_label))
        print('iteration: {}, testing pearson coefficient: {:.4}'.format(flag, p_test))
        test_pearson_pro.append(p_test)

        p_valid = pearson_coefficient(to_numpy(valid_predict), to_numpy(valid_label))
        print('iteration: {}, validation pearson coefficient: {:.4}'.format(flag, p_valid))
        valid_pearson_pro.append(p_valid)

        # compute and print the absolute error
        train_out = train_predict - train_label
        train_error = np.abs(to_numpy(train_out)).mean()
        test_out = test_predict - test_label
        test_error = np.abs(to_numpy(test_out)).mean()
        valid_out = valid_predict - valid_label
        valid_error = np.abs(to_numpy(valid_out)).mean()
        print('iteration: {}, train_error: {:.4}, test_error: {:.4}, '
              'valid_error: {:.4}'.format(flag, train_error, test_error, valid_error))

        # save model parameters
        if threshold < p_test and args.save:
            torch.save(model, pps + '/checkpoint.tar')
            threshold = p_test

print('the number of iterations: {}'.format(flag))

# save the training process
np.save(pps + "/train_loss.npy", train_pro)
np.save(pps + "/test_loss.npy", test_pro)
np.save(pps + "/valid_loss.npy", valid_pro)
np.save(pps + "/train_pearson.npy", train_pearson_pro)
np.save(pps + "/test_pearson.npy", test_pearson_pro)
np.save(pps + "/valid_pearson.npy", valid_pearson_pro)






