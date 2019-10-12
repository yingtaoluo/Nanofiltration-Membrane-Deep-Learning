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
aux_args.add_argument('--data', type=str, choices=['72_0', '72_1', '72_2', '78_0', '78_1', '78_2'],
                      help='choose to load the pre-trained parameters or randomly initialize parameters')
aux_args.add_argument('--num', type=int, help='choose the num of iteration up to which to evaluate')
aux_args.add_argument('--interval', type=int, help='choose the num of iteration up to which to evaluate')
parser.set_defaults(data='0',
                    num=2500,
                    interval=100)
args = parser.parse_args()

a = 148 if args.data[0:2] == '72' else 160

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
    train_72_0 = Data(pps, '72_0', 'train', '_loss', args.num, args.interval)
    train_72_1 = Data(pps, '72_1', 'train', '_loss', args.num, args.interval)
    train_78_0 = Data(pps, '78_0', 'train', '_loss', args.num, args.interval)
    train_78_1 = Data(pps, '78_1', 'train', '_loss', args.num, args.interval)
    test_72_0 = Data(pps, '72_0', 'test', '_loss', args.num, args.interval)
    test_72_1 = Data(pps, '72_1', 'test', '_loss', args.num, args.interval)
    test_78_0 = Data(pps, '78_0', 'test', '_loss', args.num, args.interval)
    test_78_1 = Data(pps, '78_1', 'test', '_loss', args.num, args.interval)
    valid_72_0 = Data(pps, '72_0', 'valid', '_loss', args.num, args.interval)
    valid_72_1 = Data(pps, '72_1', 'valid', '_loss', args.num, args.interval)
    valid_78_0 = Data(pps, '78_0', 'valid', '_loss', args.num, args.interval)
    valid_78_1 = Data(pps, '78_1', 'valid', '_loss', args.num, args.interval)
    p_train_72_0 = Data(pps, '72_0', 'train', '_pearson', args.num, args.interval)
    p_train_72_1 = Data(pps, '72_1', 'train', '_pearson', args.num, args.interval)
    p_train_78_0 = Data(pps, '78_0', 'train', '_pearson', args.num, args.interval)
    p_train_78_1 = Data(pps, '78_1', 'train', '_pearson', args.num, args.interval)
    p_test_72_0 = Data(pps, '72_0', 'test', '_pearson', args.num, args.interval)
    p_test_72_1 = Data(pps, '72_1', 'test', '_pearson', args.num, args.interval)
    p_test_78_0 = Data(pps, '78_0', 'test', '_pearson', args.num, args.interval)
    p_test_78_1 = Data(pps, '78_1', 'test', '_pearson', args.num, args.interval)
    p_valid_72_0 = Data(pps, '72_0', 'valid', '_pearson', args.num, args.interval)
    p_valid_72_1 = Data(pps, '72_1', 'valid', '_pearson', args.num, args.interval)
    p_valid_78_0 = Data(pps, '78_0', 'valid', '_pearson', args.num, args.interval)
    p_valid_78_1 = Data(pps, '78_1', 'valid', '_pearson', args.num, args.interval)

    # first figure
    plt.figure(1)
    plt.plot(train_72_0.index, train_72_0.item, color=color1)
    plt.plot(train_72_1.index, train_72_1.item, color=color2)
    plt.plot(train_72_0.arg_index, train_72_0.arg_item, color=color1, marker='o',
             markersize=7, label='dataset 72_0: ' + str(round(train_72_0.arg_item, 5)))
    plt.plot(train_72_1.arg_index, train_72_1.arg_item, color=color2, marker='o',
             markersize=7, label='dataset 72_1: ' + str(round(train_72_1.arg_item, 5)))

    plt.legend(loc='upper right')
    plt.title('Training Loss Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + 'train_loss_72.jpg')
    plt.close(1)

    # second figure
    plt.figure(2)
    plt.plot(test_72_0.index, test_72_0.item, color=color1)
    plt.plot(test_72_1.index, test_72_1.item, color=color2)
    plt.plot(test_72_0.arg_index, test_72_0.arg_item, color=color1, marker='o',
             markersize=7, label='dataset 72_0: ' + str(round(test_72_0.arg_item, 5)))
    plt.plot(test_72_1.arg_index, test_72_1.arg_item, color=color2, marker='o',
             markersize=7, label='dataset 72_1: ' + str(round(test_72_1.arg_item, 5)))

    plt.legend(loc='upper right')
    plt.title('Testing Loss Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + 'test_loss_72.jpg')
    plt.close(2)

    # third figure
    plt.figure(3)
    plt.plot(valid_72_0.index, valid_72_0.item, color=color1)
    plt.plot(valid_72_1.index, valid_72_1.item, color=color2)
    plt.plot(valid_72_0.arg_index, valid_72_0.arg_item, color=color1, marker='o',
             markersize=7, label='dataset 0: ' + str(round(valid_72_0.arg_item, 5)))
    plt.plot(valid_72_1.arg_index, valid_72_1.arg_item, color=color2, marker='o',
             markersize=7, label='dataset 1: ' + str(round(valid_72_1.arg_item, 5)))

    plt.legend(loc='upper right')
    plt.title('Validation Loss Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + 'valid_loss_72.jpg')
    plt.close(3)

    # fourth figure
    plt.figure(4)
    plt.plot(train_78_0.index, train_78_0.item, color=color1)
    plt.plot(train_78_1.index, train_78_1.item, color=color2)
    plt.plot(train_78_0.arg_index, train_78_0.arg_item, color=color1, marker='o',
             markersize=7, label='dataset 78_0: ' + str(round(train_78_0.arg_item, 5)))
    plt.plot(train_78_1.arg_index, train_78_1.arg_item, color=color2, marker='o',
             markersize=7, label='dataset 78_1: ' + str(round(train_78_1.arg_item, 5)))

    plt.legend(loc='upper right')
    plt.title('Training Loss Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + 'train_loss_78.jpg')
    plt.close(4)

    # fifth figure
    plt.figure(5)
    plt.plot(test_78_0.index, test_78_0.item, color=color1)
    plt.plot(test_78_1.index, test_78_1.item, color=color2)
    plt.plot(test_78_0.arg_index, test_78_0.arg_item, color=color1, marker='o',
             markersize=7, label='dataset 78_0: ' + str(round(test_78_0.arg_item, 5)))
    plt.plot(test_78_1.arg_index, test_78_1.arg_item, color=color2, marker='o',
             markersize=7, label='dataset 78_1: ' + str(round(test_78_1.arg_item, 5)))

    plt.legend(loc='upper right')
    plt.title('Testing Loss Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + 'test_loss_78.jpg')
    plt.close(5)

    # sixth figure
    plt.figure(6)
    plt.plot(valid_78_0.index, valid_78_0.item, color=color1)
    plt.plot(valid_78_1.index, valid_78_1.item, color=color2)
    plt.plot(valid_78_0.arg_index, valid_78_0.arg_item, color=color1, marker='o',
             markersize=7, label='dataset 78_0: ' + str(round(valid_78_0.arg_item, 5)))
    plt.plot(valid_78_1.arg_index, valid_78_1.arg_item, color=color2, marker='o',
             markersize=7, label='dataset 78_1: ' + str(round(valid_78_1.arg_item, 5)))

    plt.legend(loc='upper right')
    plt.title('Validation Loss Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + 'valid_loss_78.jpg')
    plt.close(6)

    # first figure
    plt.figure(1)
    plt.plot(p_train_72_0.index, p_train_72_0.item, color=color1)
    plt.plot(p_train_72_1.index, p_train_72_1.item, color=color2)
    plt.plot(p_train_72_0.arg_index, p_train_72_0.arg_item, color=color1, marker='o',
             markersize=7, label='dataset 72_0: ' + str(round(p_train_72_0.arg_item, 5)))
    plt.plot(p_train_72_1.arg_index, p_train_72_1.arg_item, color=color2, marker='o',
             markersize=7, label='dataset 72_1: ' + str(round(p_train_72_1.arg_item, 5)))

    plt.legend(loc='lower right')
    plt.title('Training Pearson Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + 'train_pearson_72.jpg')
    plt.close(1)

    # second figure
    plt.figure(2)
    plt.plot(p_test_72_0.index, p_test_72_0.item, color=color1)
    plt.plot(p_test_72_1.index, p_test_72_1.item, color=color2)
    plt.plot(p_test_72_0.arg_index, p_test_72_0.arg_item, color=color1, marker='o',
             markersize=7, label='dataset 72_0: ' + str(round(p_test_72_0.arg_item, 5)))
    plt.plot(p_test_72_1.arg_index, p_test_72_1.arg_item, color=color2, marker='o',
             markersize=7, label='dataset 72_1: ' + str(round(p_test_72_1.arg_item, 5)))

    plt.legend(loc='lower right')
    plt.title('Testing Pearson Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + 'test_pearson_72.jpg')
    plt.close(2)

    # third figure
    plt.figure(3)
    plt.plot(p_valid_72_0.index, p_valid_72_0.item, color=color1)
    plt.plot(p_valid_72_1.index, p_valid_72_1.item, color=color2)
    plt.plot(p_valid_72_0.arg_index, p_valid_72_0.arg_item, color=color1, marker='o',
             markersize=7, label='dataset 72_0: ' + str(round(p_valid_72_0.arg_item, 5)))
    plt.plot(p_valid_72_1.arg_index, p_valid_72_1.arg_item, color=color2, marker='o',
             markersize=7, label='dataset 72_1: ' + str(round(p_valid_72_1.arg_item, 5)))

    plt.legend(loc='lower right')
    plt.title('Validation Pearson Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + 'valid_pearson_72.jpg')
    plt.close(3)

    # fourth figure
    plt.figure(4)
    plt.plot(p_train_78_0.index, p_train_78_0.item, color=color1)
    plt.plot(p_train_78_1.index, p_train_78_1.item, color=color2)
    plt.plot(p_train_78_0.arg_index, p_train_78_0.arg_item, color=color1, marker='o',
             markersize=7, label='dataset 78_0: ' + str(round(p_train_78_0.arg_item, 5)))
    plt.plot(p_train_78_1.arg_index, p_train_78_1.arg_item, color=color2, marker='o',
             markersize=7, label='dataset 78_1: ' + str(round(p_train_78_1.arg_item, 5)))

    plt.legend(loc='lower right')
    plt.title('Training Pearson Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + 'train_pearson_78.jpg')
    plt.close(4)

    # fifth figure
    plt.figure(5)
    plt.plot(p_test_78_0.index, p_test_78_0.item, color=color1)
    plt.plot(p_test_78_1.index, p_test_78_1.item, color=color2)
    plt.plot(p_test_78_0.arg_index, p_test_78_0.arg_item, color=color1, marker='o',
             markersize=7, label='dataset 78_0: ' + str(round(p_test_78_0.arg_item, 5)))
    plt.plot(p_test_78_1.arg_index, p_test_78_1.arg_item, color=color2, marker='o',
             markersize=7, label='dataset 78_1: ' + str(round(p_test_78_1.arg_item, 5)))

    plt.legend(loc='lower right')
    plt.title('Testing Pearson Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + 'test_pearson_78.jpg')
    plt.close(5)

    # sixth figure
    plt.figure(6)
    plt.plot(p_valid_78_0.index, p_valid_78_0.item, color=color1)
    plt.plot(p_valid_78_1.index, p_valid_78_1.item, color=color2)
    plt.plot(p_valid_78_0.arg_index, p_valid_78_0.arg_item, color=color1, marker='o',
             markersize=7, label='dataset 78_0: ' + str(round(p_valid_78_0.arg_item, 5)))
    plt.plot(p_valid_78_1.arg_index, p_valid_78_1.arg_item, color=color2, marker='o',
             markersize=7, label='dataset 78_1: ' + str(round(p_valid_78_1.arg_item, 5)))

    plt.legend(loc='lower right')
    plt.title('Validation Pearson Evaluation', fontsize=font1)
    plt.xlabel('number of iterations', fontsize=font2)
    plt.ylabel('value', fontsize=font2)
    plt.grid(ls='--')
    plt.savefig(sfp + args.choice + '/' + args.type + '/' + 'valid_pearson_78.jpg')
    plt.close(6)

elif args.type == 'correlation':
    # data directory
    dps = './data/' + args.data
    train_data = np.load(dps + "/train_dataset.npy")
    test_data = np.load(dps + "/test_dataset.npy")
    valid_data = np.load(dps + "/valid_dataset.npy")

    a = 148 if args.data[0:2] == '72' else 160
    train_input = Variable(torch.FloatTensor(train_data[:, 0:a]), requires_grad=False)
    if args.choice == 'rejection':
        train_label = train_data[:, a+1:a+2]
    elif args.choice == 'permeability':
        train_label = train_data[:, a:a+1]
    else:
        raise NotImplementedError()
    test_input = Variable(torch.FloatTensor(test_data[:, 0:a]), requires_grad=False)
    if args.choice == 'rejection':
        test_label = test_data[:, a+1:a+2]
    elif args.choice == 'permeability':
        test_label = test_data[:, a:a+1]
    else:
        raise NotImplementedError()
    valid_input = Variable(torch.FloatTensor(valid_data[:, 0:a]), requires_grad=False)
    if args.choice == 'rejection':
        valid_label = valid_data[:, a+1:a+2]
    elif args.choice == 'permeability':
        valid_label = valid_data[:, a:a+1]
    else:
        raise NotImplementedError()

    model = neural_network(choice=args.choice, init='load', data=args.data, device='cpu')
    train_predict = model(train_input)
    test_predict = model(test_input)
    valid_predict = model(valid_input)

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
