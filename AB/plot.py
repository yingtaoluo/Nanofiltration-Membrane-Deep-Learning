import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import argparse
from model import neural_network


color1 = 'goldenrod'
color2 = 'black'
color3 = 'royalblue'
color4 = 'orange'
color5 = 'navy'

font1 = 25
font2 = 15

font = {'family': 'Times New Roman',
        'size': 15}

plt.rc('font', **font)
sfp = './figures/'


# truncate the file with a certain interval
def loader(file, interval, index):
    vec = []
    it = []
    for i, content in enumerate(file):
        if i % interval == 0 or i == index:
            vec.append(content)
            it.append(i)
    vec = np.array(vec)
    it = np.array(it)
    return it, vec


class Data:
    item = 0
    index = 0
    arg_index = 0
    arg_item = 0

    def __init__(self, directory, data, state1, state2, num, interval):
        file = np.load(directory + '/' + data + '/' + state1 + state2 + '.npy')[0:num]
        if state2 == '_pearson':
            self.arg_index = int(np.argmax(file))
        elif state2 == '_loss':
            self.arg_index = int(np.argmin(file))
        self.arg_item = file[self.arg_index]
        self.index, self.item = loader(file, interval, self.arg_index)

    def print(self):
        print("The items of the data are: {}".format(self.item))
        print("The indexes of the data are: {}".format(self.index))


'''command prompt resolution'''
parser = argparse.ArgumentParser()
parser.add_argument('choice', type=str, choices=['rejection', 'permeability'],
                    help='choose to predict rejection or permeability')
parser.add_argument('type', type=str, choices=['alone', 'together', 'correlation'],
                    help='choose to print the figure per data, of all data, or the correlation')
# other auxiliary factors
aux_args = parser.add_argument_group('auxiliary')
aux_args.add_argument('--data', type=str, choices=['0', '1', '2', '3', '4'],
                      help='choose to load the pre-trained parameters or randomly initialize parameters')
aux_args.add_argument('--num', type=int, help='choose the num of iteration up to which to evaluate')
aux_args.add_argument('--interval', type=int, help='choose the num of iteration up to which to evaluate')
parser.set_defaults(data='0',
                    num=2500,
                    interval=100)
args = parser.parse_args()


if args.type == 'alone':
    # loading the data
    pps = './process/' + args.choice
    train = Data(pps, args.data, 'train', '_loss', args.num, args.interval)
    test = Data(pps, args.data, 'test', '_loss', args.num, args.interval)
    valid = Data(pps, args.data, 'valid', '_loss', args.num, args.interval)
    p_train = Data(pps, args.data, 'train', '_pearson', args.num, args.interval)
    p_test = Data(pps, args.data, 'test', '_pearson', args.num, args.interval)
    p_valid = Data(pps, args.data, 'valid', '_pearson', args.num, args.interval)

    # find the index of the min / max value in the file
    train_index = int(np.argmin(train.arg_index))
    test_index = int(np.argmin(test.arg_index))
    valid_index = int(np.argmin(valid.arg_index))
    p_train_index = int(np.argmax(p_train.arg_index))
    p_test_index = int(np.argmax(p_test.arg_index))
    p_valid_index = int(np.argmax(p_valid.arg_index))

    # auxiliary operations
    a = np.linspace(0, args.num - 1, args.num)
    b = np.linspace(0, args.num - 1, args.num)

    # first figure
    plt.figure(1)
    plt.plot(train.index, train.item, color='goldenrod', label='training loss')
    plt.plot(test.index, test.item, color='black', label='testing loss')
    plt.plot(valid.index, valid.item, color='royalblue', label='validation loss')

    plt.plot(train.arg_index, train.arg_item, color=color1, marker='o',
             markersize=7, label='minimum: ' + str(round(train.arg_item, 5)))
    plt.plot(test.arg_index, test.arg_item, color=color2, marker='o',
             markersize=7, label='minimum: ' + str(round(test.arg_item, 5)))
    plt.plot(valid.arg_index, valid.arg_item, color=color3, marker='o',
             markersize=7, label='minimum: ' + str(round(valid.arg_item, 5)))

    plt.legend(loc='upper right')
    plt.title('Loss Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.tick_params(labelsize=font2)
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + args.data + '/' + 'loss.jpg')

    # second figure
    plt.figure(2)
    plt.plot(p_train.index, p_train.item, color='goldenrod', label='training pearson coefficient')
    plt.plot(p_test.index, p_test.item, color='black', label='testing pearson coefficient')
    plt.plot(p_valid.index, p_valid.item, color='royalblue', label='validation pearson coefficient')

    plt.plot(p_train.arg_index, p_train.arg_item, color=color1, marker='o',
             markersize=7, label='maximum: ' + str(round(p_train.arg_item, 5)))
    plt.plot(p_test.arg_index, p_test.arg_item, color=color2, marker='o',
             markersize=7, label='maximum: ' + str(round(p_test.arg_item, 5)))
    plt.plot(p_valid.arg_index, p_valid.arg_item, color=color3, marker='o',
             markersize=7, label='maximum: ' + str(round(p_valid.arg_item, 5)))

    plt.legend(loc='lower right')
    plt.title('Pearson Coefficient Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + args.data + '/' + 'pearson.jpg')

    # plt.show()


elif args.type == 'together':
    # loading the data
    pps = './process/' + args.choice
    train_0 = Data(pps, '0', 'train', '_loss', args.num, args.interval)
    train_1 = Data(pps, '1', 'train', '_loss', args.num, args.interval)
    train_2 = Data(pps, '2', 'train', '_loss', args.num, args.interval)
    train_3 = Data(pps, '3', 'train', '_loss', args.num, args.interval)
    train_4 = Data(pps, '4', 'train', '_loss', args.num, args.interval)
    test_0 = Data(pps, '0', 'test', '_loss', args.num, args.interval)
    test_1 = Data(pps, '1', 'test', '_loss', args.num, args.interval)
    test_2 = Data(pps, '2', 'test', '_loss', args.num, args.interval)
    test_3 = Data(pps, '3', 'test', '_loss', args.num, args.interval)
    test_4 = Data(pps, '4', 'test', '_loss', args.num, args.interval)
    valid_0 = Data(pps, '0', 'valid', '_loss', args.num, args.interval)
    valid_1 = Data(pps, '1', 'valid', '_loss', args.num, args.interval)
    valid_2 = Data(pps, '2', 'valid', '_loss', args.num, args.interval)
    valid_3 = Data(pps, '3', 'valid', '_loss', args.num, args.interval)
    valid_4 = Data(pps, '4', 'valid', '_loss', args.num, args.interval)
    p_train_0 = Data(pps, '0', 'train', '_pearson', args.num, args.interval)
    p_train_1 = Data(pps, '1', 'train', '_pearson', args.num, args.interval)
    p_train_2 = Data(pps, '2', 'train', '_pearson', args.num, args.interval)
    p_train_3 = Data(pps, '3', 'train', '_pearson', args.num, args.interval)
    p_train_4 = Data(pps, '4', 'train', '_pearson', args.num, args.interval)
    p_test_0 = Data(pps, '0', 'test', '_pearson', args.num, args.interval)
    p_test_1 = Data(pps, '1', 'test', '_pearson', args.num, args.interval)
    p_test_2 = Data(pps, '2', 'test', '_pearson', args.num, args.interval)
    p_test_3 = Data(pps, '3', 'test', '_pearson', args.num, args.interval)
    p_test_4 = Data(pps, '4', 'test', '_pearson', args.num, args.interval)
    p_valid_0 = Data(pps, '0', 'valid', '_pearson', args.num, args.interval)
    p_valid_1 = Data(pps, '1', 'valid', '_pearson', args.num, args.interval)
    p_valid_2 = Data(pps, '2', 'valid', '_pearson', args.num, args.interval)
    p_valid_3 = Data(pps, '3', 'valid', '_pearson', args.num, args.interval)
    p_valid_4 = Data(pps, '4', 'valid', '_pearson', args.num, args.interval)

    # first figure
    fig1 = plt.figure(1)
    plt.plot(train_0.index, train_0.item, color=color1)
    plt.plot(train_1.index, train_1.item, color=color2)
    plt.plot(train_2.index, train_2.item, color=color3)
    plt.plot(train_3.index, train_3.item, color=color4)
    plt.plot(train_4.index, train_4.item, color=color5)

    plt.plot(train_0.arg_index, train_0.arg_item, color=color1, marker='o',
             markersize=7, label='dataset 0: ' + str(round(train_0.arg_item, 5)))
    plt.plot(train_1.arg_index, train_1.arg_item, color=color2, marker='o',
             markersize=7, label='dataset 1: ' + str(round(train_1.arg_item, 5)))
    plt.plot(train_2.arg_index, train_2.arg_item, color=color3, marker='o',
             markersize=7, label='dataset 2: ' + str(round(train_2.arg_item, 5)))
    plt.plot(train_3.arg_index, train_3.arg_item, color=color4, marker='o',
             markersize=7, label='dataset 3: ' + str(round(train_3.arg_item, 5)))
    plt.plot(train_4.arg_index, train_4.arg_item, color=color5, marker='o',
             markersize=7, label='dataset 4: ' + str(round(train_4.arg_item, 5)))

    plt.legend(loc='upper right')
    plt.title('Training Loss Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + 'train_loss.jpg')

    # second figure
    plt.figure(2)
    plt.plot(test_0.index, test_0.item, color=color1)
    plt.plot(test_1.index, test_1.item, color=color2)
    plt.plot(test_2.index, test_2.item, color=color3)
    plt.plot(test_3.index, test_3.item, color=color4)
    plt.plot(test_4.index, test_4.item, color=color5)

    plt.plot(test_0.arg_index, test_0.arg_item, color=color1, marker='o',
             markersize=7, label='dataset 0: ' + str(round(test_0.arg_item, 5)))
    plt.plot(test_1.arg_index, test_1.arg_item, color=color2, marker='o',
             markersize=7, label='dataset 1: ' + str(round(test_1.arg_item, 5)))
    plt.plot(test_2.arg_index, test_2.arg_item, color=color3, marker='o',
             markersize=7, label='dataset 2: ' + str(round(test_2.arg_item, 5)))
    plt.plot(test_3.arg_index, test_3.arg_item, color=color4, marker='o',
             markersize=7, label='dataset 3: ' + str(round(test_3.arg_item, 5)))
    plt.plot(test_4.arg_index, test_4.arg_item, color=color5, marker='o',
             markersize=7, label='dataset 4: ' + str(round(test_4.arg_item, 5)))

    plt.legend(loc='upper right')
    plt.title('Testing Loss Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + 'test_loss.jpg')

    # third figure
    plt.figure(3)
    plt.plot(valid_0.index, valid_0.item, color=color1)
    plt.plot(valid_1.index, valid_1.item, color=color2)
    plt.plot(valid_2.index, valid_2.item, color=color3)
    plt.plot(valid_3.index, valid_3.item, color=color4)
    plt.plot(valid_4.index, valid_4.item, color=color5)

    plt.plot(valid_0.arg_index, valid_0.arg_item, color=color1, marker='o',
             markersize=7, label='dataset 0: ' + str(round(valid_0.arg_item, 5)))
    plt.plot(valid_1.arg_index, valid_1.arg_item, color=color2, marker='o',
             markersize=7, label='dataset 1: ' + str(round(valid_1.arg_item, 5)))
    plt.plot(valid_2.arg_index, valid_2.arg_item, color=color3, marker='o',
             markersize=7, label='dataset 2: ' + str(round(valid_2.arg_item, 5)))
    plt.plot(valid_3.arg_index, valid_3.arg_item, color=color4, marker='o',
             markersize=7, label='dataset 3: ' + str(round(valid_3.arg_item, 5)))
    plt.plot(valid_4.arg_index, valid_4.arg_item, color=color5, marker='o',
             markersize=7, label='dataset 4: ' + str(round(valid_4.arg_item, 5)))

    plt.legend(loc='upper right')
    plt.title('Validation Loss Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + 'valid_loss.jpg')

    # fourth figure
    plt.figure(4)
    plt.plot(p_train_0.index, p_train_0.item, color=color1)
    plt.plot(p_train_1.index, p_train_1.item, color=color2)
    plt.plot(p_train_2.index, p_train_2.item, color=color3)
    plt.plot(p_train_3.index, p_train_3.item, color=color4)
    plt.plot(p_train_4.index, p_train_4.item, color=color5)

    plt.plot(p_train_0.arg_index, p_train_0.arg_item, color=color1, marker='o',
             markersize=7, label='dataset 0: ' + str(round(p_train_0.arg_item, 5)))
    plt.plot(p_train_1.arg_index, p_train_1.arg_item, color=color2, marker='o',
             markersize=7, label='dataset 1: ' + str(round(p_train_1.arg_item, 5)))
    plt.plot(p_train_2.arg_index, p_train_2.arg_item, color=color3, marker='o',
             markersize=7, label='dataset 2: ' + str(round(p_train_2.arg_item, 5)))
    plt.plot(p_train_3.arg_index, p_train_3.arg_item, color=color4, marker='o',
             markersize=7, label='dataset 3: ' + str(round(p_train_3.arg_item, 5)))
    plt.plot(p_train_4.arg_index, p_train_4.arg_item, color=color5, marker='o',
             markersize=7, label='dataset 4: ' + str(round(p_train_4.arg_item, 5)))

    plt.legend(loc='lower right')
    plt.title('Training Pearson Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + 'train_pearson.jpg')

    # fifth figure
    plt.figure(5)
    plt.plot(p_test_0.index, p_test_0.item, color=color1)
    plt.plot(p_test_1.index, p_test_1.item, color=color2)
    plt.plot(p_test_2.index, p_test_2.item, color=color3)
    plt.plot(p_test_3.index, p_test_3.item, color=color4)
    plt.plot(p_test_4.index, p_test_4.item, color=color5)

    plt.plot(p_test_0.arg_index, p_test_0.arg_item, color=color1, marker='o',
             markersize=7, label='dataset 0: ' + str(round(p_test_0.arg_item, 5)))
    plt.plot(p_test_1.arg_index, p_test_1.arg_item, color=color2, marker='o',
             markersize=7, label='dataset 1: ' + str(round(p_test_1.arg_item, 5)))
    plt.plot(p_test_2.arg_index, p_test_2.arg_item, color=color3, marker='o',
             markersize=7, label='dataset 2: ' + str(round(p_test_2.arg_item, 5)))
    plt.plot(p_test_3.arg_index, p_test_3.arg_item, color=color4, marker='o',
             markersize=7, label='dataset 3: ' + str(round(p_test_3.arg_item, 5)))
    plt.plot(p_test_4.arg_index, p_test_4.arg_item, color=color5, marker='o',
             markersize=7, label='dataset 4: ' + str(round(p_test_4.arg_item, 5)))

    plt.legend(loc='lower right')
    plt.title('Testing Pearson Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + 'test_pearson.jpg')

    # sixth figure
    plt.figure(6)
    plt.plot(p_valid_0.index, p_valid_0.item, color=color1)
    plt.plot(p_valid_1.index, p_valid_1.item, color=color2)
    plt.plot(p_valid_2.index, p_valid_2.item, color=color3)
    plt.plot(p_valid_3.index, p_valid_3.item, color=color4)
    plt.plot(p_valid_4.index, p_valid_4.item, color=color5)

    plt.plot(p_valid_0.arg_index, p_valid_0.arg_item, color=color1, marker='o',
             markersize=7, label='dataset 0: ' + str(round(p_valid_0.arg_item, 5)))
    plt.plot(p_valid_1.arg_index, p_valid_1.arg_item, color=color2, marker='o',
             markersize=7, label='dataset 1: ' + str(round(p_valid_1.arg_item, 5)))
    plt.plot(p_valid_2.arg_index, p_valid_2.arg_item, color=color3, marker='o',
             markersize=7, label='dataset 2: ' + str(round(p_valid_2.arg_item, 5)))
    plt.plot(p_valid_3.arg_index, p_valid_3.arg_item, color=color4, marker='o',
             markersize=7, label='dataset 3: ' + str(round(p_valid_3.arg_item, 5)))
    plt.plot(p_valid_4.arg_index, p_valid_4.arg_item, color=color5, marker='o',
             markersize=7, label='dataset 4: ' + str(round(p_valid_4.arg_item, 5)))

    plt.legend(loc='lower right')
    plt.title('Validation Pearson Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + 'valid_pearson.jpg')

    # plt.show()

elif args.type == 'correlation':
    # data directory
    dps = './data/' + args.data
    train_data = np.load(dps + "/train_data.npy")  # (320000,446)
    test_data = np.load(dps + "/test_data.npy")  # (29290, 446)
    valid_data = np.load(dps + "/valid_data.npy")  # (70700, 446)
    train_input = Variable(torch.FloatTensor(train_data[:, 0:444]), requires_grad=False)
    if args.choice == 'rejection':
        train_label = train_data[:, 445:446]
    elif args.choice == 'permeability':
        train_label = train_data[:, 444:445] * 100
    else:
        raise NotImplementedError()
    test_input = Variable(torch.FloatTensor(test_data[:, 0:444]), requires_grad=False)
    if args.choice == 'rejection':
        test_label = test_data[:, 445:446]
    elif args.choice == 'permeability':
        test_label = test_data[:, 444:445] * 100
    else:
        raise NotImplementedError()
    valid_input = Variable(torch.FloatTensor(valid_data[:, 0:444]), requires_grad=False)
    if args.choice == 'rejection':
        valid_label = valid_data[:, 445:446]
    elif args.choice == 'permeability':
        valid_label = valid_data[:, 444:445] * 100
    else:
        raise NotImplementedError()

    model = neural_network(choice=args.choice, init='load', data=args.data, device='cpu')
    train_predict = model(train_input)
    test_predict = model(test_input)
    valid_predict = model(valid_input)
    if args.choice == 'permeability':
        train_predict *= 100
        test_predict *= 100
        valid_predict *= 100

    # the first figure
    plt.figure(1)
    plt.plot(train_label, train_label, color=color1, linestyle=":",
             marker='^', markersize=5, label='real')
    plt.scatter(train_label, train_predict.detach().numpy(),
                c=color3, s=10, marker='v', label='predict')
    plt.legend(loc='lower right')
    plt.title('Training Correlation Evaluation', fontsize=font1)
    if args.choice == 'rejection':
        plt.xlabel('real rejection rate', fontsize=font2)
        plt.ylabel('predict / real rejection rate', fontsize=font2)
    elif args.choice == 'permeability':
        plt.xlabel('real flux: L/(h·bar·m^2)', fontsize=font2)
        plt.ylabel('predict / real flux: L/(h·bar·m^2)', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + args.data + '/' + 'train.jpg')

    # the second figure
    plt.figure(2)
    plt.plot(test_label, test_label, color=color1, linestyle=":",
             marker='^', markersize=5, label='real')
    plt.scatter(test_label, test_predict.detach().numpy(),
                c=color3, s=10, marker='o', label='predict')
    plt.legend(loc='lower right')
    plt.title('Testing Correlation Evaluation', fontsize=font1)
    if args.choice == 'rejection':
        plt.xlabel('real rejection rate', fontsize=font2)
        plt.ylabel('predict / real rejection rate', fontsize=font2)
    elif args.choice == 'permeability':
        plt.xlabel('real flux: L/(h·bar·m^2)', fontsize=font2)
        plt.ylabel('predict / real flux: L/(h·bar·m^2)', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + args.data + '/' + 'test.jpg')

    # the third figure
    plt.figure(3)
    plt.plot(valid_label, valid_label, color=color1, linestyle=":",
             marker='^', markersize=5, label='real')
    plt.scatter(valid_label, valid_predict.detach().numpy(),
                c=color3, s=10, marker='o', label='predict')
    plt.legend(loc='lower right')
    plt.title('Validation Correlation Evaluation', fontsize=font1)
    if args.choice == 'rejection':
        plt.xlabel('real rejection rate', fontsize=font2)
        plt.ylabel('predict / real rejection rate', fontsize=font2)
    elif args.choice == 'permeability':
        plt.xlabel('real flux: L/(h·bar·m^2)', fontsize=font2)
        plt.ylabel('predict / real flux: L/(h·bar·m^2)', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + args.data + '/' + 'valid.jpg')

    # plt.show()
