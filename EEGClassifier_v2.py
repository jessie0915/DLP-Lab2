import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
from torchsummary import summary
from dataloader_v1 import bci_train_Dataset, bci_test_Dataset
from torch.utils.data import DataLoader

# use cuda or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 300


class EEGNet(nn.Module):
    def __init__(self, activation_type):
        super(EEGNet, self).__init__()

        # Layer 1 : first convolution
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Layer 2 : depthWiseConv
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if activation_type == 0:
            self.activation2 = nn.ELU(alpha=1.0)
        elif activation_type == 1:
            self.activation2 = nn.LeakyReLU()
        elif activation_type == 2:
            self.activation2 = nn.ReLU()
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0)
        self.dropout2 = nn.Dropout(p=0.5)

        # Layer 3 : separableConv
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1),  padding=(0, 7), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if activation_type == 0:
            self.activation3 = nn.ELU(alpha=1.0)
        elif activation_type == 1:
            self.activation3 = nn.LeakyReLU()
        elif activation_type == 2:
            self.activation3 = nn.ReLU()
        self.pooling3 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0)
        self.dropout3 = nn.Dropout(p=0.5)

        # classify
        self.fc1 = nn.Linear(in_features=736, out_features=2, bias=True)

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.batchnorm1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        # Layer 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.activation3(x)
        x = self.pooling3(x)
        x = self.dropout3(x)

        # classify
        x = x.view(-1, 736)
        x = self.fc1(x)

        return x


class DeepConvNet(nn.Module):
    def __init__(self, activation_type):
        super(DeepConvNet, self).__init__()

        # output_shape = (image_shape-filter_shape+2*padding)/stride + 1
        # Layer 1 : first 2 convolutions
        self.conv1_1 = nn.Conv2d(1, 25, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False)
        self.conv1_2 = nn.Conv2d(25, 25, kernel_size=(2, 1), stride=(1, 1), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if activation_type == 0:
            self.activation1 = nn.ELU(alpha=1.0)
        elif activation_type == 1:
            self.activation1 = nn.LeakyReLU()
        elif activation_type == 2:
            self.activation1 = nn.ReLU()
        self.pooling1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)
        self.dropout1 = nn.Dropout(p=0.5)

        # Layer 2 : second convolution
        self.conv2 = nn.Conv2d(25, 50, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if activation_type == 0:
            self.activation2 = nn.ELU(alpha=1.0)
        elif activation_type == 1:
            self.activation2 = nn.LeakyReLU()
        elif activation_type == 2:
            self.activation2 = nn.ReLU()
        self.pooling2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)
        self.dropout2 = nn.Dropout(p=0.5)

        # Layer 3 : third convolution
        self.conv3 = nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1, 1),  padding=(0, 2), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if activation_type == 0:
            self.activation3 = nn.ELU(alpha=1.0)
        elif activation_type == 1:
            self.activation3 = nn.LeakyReLU()
        elif activation_type == 2:
            self.activation3 = nn.ReLU()
        self.pooling3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)
        self.dropout3 = nn.Dropout(p=0.5)

        # Layer 4 : fourth convolution
        self.conv4 = nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False)
        self.batchnorm4 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if activation_type == 0:
            self.activation4 = nn.ELU(alpha=1.0)
        elif activation_type == 1:
            self.activation4 = nn.LeakyReLU()
        elif activation_type == 2:
            self.activation4 = nn.ReLU()
        self.pooling4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)
        self.dropout4 = nn.Dropout(p=0.5)

        # classify
        self.fc1 = nn.Linear(in_features=9200, out_features=2, bias=True)

    def forward(self, x):
        # Layer 1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.batchnorm1(x)
        x = self.activation1(x)
        x = self.pooling1(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        # Layer 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.activation3(x)
        x = self.pooling3(x)
        x = self.dropout3(x)

        # Layer 4
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.activation4(x)
        x = self.pooling4(x)
        x = self.dropout4(x)

        # classify
        x = x.view(-1, 9200)
        x = self.fc1(x)

        return x


'''
Testing function
args : 
    model: the network you built, e.g., EEGNet, DeepConvNet
    data_loader: propose (batch_size, (input, label)) data
'''
def evaluate(model, data_loader):
    model.eval()

    correct = 0
    for batch_idx, (X,Y) in enumerate(data_loader):
        inputs = X.to(device)
        labels = Y.to(device)
        # wrap them in Variable
        inputs = Variable(inputs.float())
        labels = Variable(labels.long())
        with torch.no_grad():
            output = model(inputs)
        pred = torch.max(output, 1)[1]
        correct += pred.eq(labels.data).cpu().sum()

    results = correct.cpu().numpy() * 100.0 / len(data_loader.dataset)

    return results


'''
Training function
args : 
    model: the network you built, e.g., EEGNet, DeepConvNet 
    optimizer: the optimizer you set, e.g., Adam, SGD.
    criterion: the loss function you set, e.g., CrossEntropyLoss
    model_save_path:  save file name string before "best.plk"
'''
def train(model, optimizer, criterion, model_save_path):

    epoch_hist = []
    acc_train_hist = []
    acc_test_hist = []

    test_acc_max = 0.0
    if not os.path.exists('backup'):
        os.mkdir('backup')

    # training
    for epoch in range(EPOCH):
        model.train()

        running_loss = 0.0
        for batch_idx, (X, Y) in enumerate(train_loader):
            inputs = X.to(device)
            labels = Y.to(device)
            # wrap them in Variable
            inputs, labels = Variable(inputs.float()), Variable(labels.long())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Verification
        train_acc = evaluate(model, train_loader)
        test_acc = evaluate(model, test_loader)
        print('\nEpoch %d, Training Loss: %8f, Train Accuracy: %2f, Test Accuracy: %2f' % (epoch, running_loss, train_acc, test_acc))

        epoch_hist.append(epoch)
        acc_train_hist.append(train_acc)
        acc_test_hist.append(test_acc)

        if test_acc_max < test_acc:
            model_save_full_path = model_save_path + 'best.pkl'
            torch.save(model.state_dict(), model_save_full_path)
            test_acc_max = test_acc

    return epoch_hist, acc_train_hist, acc_test_hist


'''
Load dataset and create data loader
'''
trainset = bci_train_Dataset()
testset = bci_test_Dataset()
# create train/test loaders
train_loader = DataLoader(dataset=trainset,
                          batch_size=256,
                          shuffle=True,
                          num_workers=0)
test_loader = DataLoader(dataset=testset,
                         batch_size=256,
                         shuffle=False,
                         num_workers=0)

'''
demo from the best model (EEG + ReLU)
args : 
    actvation_type: 0 : ELU, 1: Leaky ReLU, 2: ReLU
    batch_size: the number of data fed to train per epoch 
    path:  save file name string before "best.plk"
'''
def demo_from_best_model(actvation_type, batch_size, path):
    net_best = EEGNet(actvation_type)
    net_best = net_best.to(device)
    net_best.load_state_dict(torch.load(path))
    net_best.eval()
    best_acc = evaluate(net_best, test_loader)
    print('test_best_accuracy = %.2f' % best_acc)


'''
training demo for one model with one Activation
args : 
    model_type: 'EEGNet' or 'DeepConvNet'
    actvation_type: 0 : ELU, 1: Leaky ReLU, 2: ReLU
    loop_number: the number of training times (not epoch)
'''
def demo(model_type, actvation_type, loop_number):

    # for saving accuracy in each loop
    train_max = np.zeros(shape=[loop_number], dtype=float)
    test_max = np.zeros(shape=[loop_number], dtype=float)
    # learning rate setting
    LR = 1e-3
    for loop in range(loop_number):
        plt.figure()
        # EEGNet Starting
        if model_type == 'EEGNet':
            w_decay = 0.01
            # set model save path
            model_save_path = 'backup/EEGNet_loop_' + str(loop)
            # Accordin
            if actvation_type == 0:
                model_save_path_a = model_save_path + '_ELU_'
                train_label_name = 'elu_train'
                test_label_name = 'elu_test'
            elif actvation_type == 1:
                model_save_path_a = model_save_path + '_LeakyReLU_'
                train_label_name = 'leaky_relu_train'
                test_label_name = 'leaky_relu_test'
            elif actvation_type == 2:
                model_save_path_a = model_save_path + '_ReLU_'
                train_label_name = 'relu_train'
                test_label_name = 'relu_test'
            else:
                model_save_path_a = model_save_path + '_Unknown_'
                train_label_name = 'Unknown_train'
                test_label_name = 'Unknown_test'

            net = EEGNet(actvation_type)
            net = net.to(device)
            summary(net, [(1, 2, 750)])
            optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=w_decay)
            criterion = nn.CrossEntropyLoss().to(device)

            epoch_hist, acc_train_hist, acc_test_hist = train(net, optimizer, criterion, model_save_path_a)
            train_max[loop] = np.max(acc_train_hist)
            test_max[loop] = np.max(acc_test_hist)

            plt.plot(epoch_hist, acc_train_hist, label=train_label_name)
            plt.plot(epoch_hist, acc_test_hist, label=test_label_name)
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy(%)")
            plt.title("Activation function comparison(EEGNet)")
            plt.legend()

        elif model_type == 'DeepConvNet':
            w_decay = 0.1
            model_save_path = 'backup/DeepConvNet_loop_' + str(loop)
            if actvation_type == 0:
                model_save_path_a = model_save_path + '_ELU_'
                train_label_name = 'elu_train'
                test_label_name = 'elu_test'
            elif actvation_type == 1:
                model_save_path_a = model_save_path + '_LeakyReLU_'
                train_label_name = 'leaky_relu_train'
                test_label_name = 'leaky_relu_test'
            elif actvation_type == 2:
                model_save_path_a = model_save_path + '_ReLU_'
                train_label_name = 'relu_train'
                test_label_name = 'relu_test'
            else:
                model_save_path_a = model_save_path + '_Unknown_'
                train_label_name = 'Unknown_train'
                test_label_name = 'Unknown_test'

            net = DeepConvNet(actvation_type)
            net = net.to(device)
            summary(net, [(1, 2, 750)])
            optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=w_decay)
            criterion = nn.CrossEntropyLoss().to(device)

            epoch_hist, acc_train_hist, acc_test_hist = train(net, optimizer, criterion, model_save_path_a)
            train_max[loop] = np.max(acc_train_hist)
            test_max[loop] = np.max(acc_test_hist)

            plt.plot(epoch_hist, acc_train_hist, label=train_label_name)
            plt.plot(epoch_hist, acc_test_hist, label=test_label_name)
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy(%)")
            plt.title("Activation function comparison(DeepConvNet)")
            plt.legend()

        else:
            print('model_type should be either EEGNet or DeepConvNet')
            return

    print(train_max)
    print(test_max)
    plt.show()


'''
training demo for one model with three Activation
Note: in this function you need to change model manually
'''
def train_for_3_activation():
    for loop in range(1):
        plt.figure()
        train_max = np.zeros(3, dtype=float)
        test_max = np.zeros(3, dtype=float)
        LR = 1e-3
        w_decay = 0.01
        for i in range(3):
            model_save_path = 'backup/EEGNet_loop_' + str(loop)
            # model_save_path = 'backup/DeepConvNet_loop_' + str(loop)
            if i == 0:
                model_save_path_a = model_save_path + '_ELU_'
                train_label_name = 'elu_train'
                test_label_name = 'elu_test'
            elif i == 1:
                model_save_path_a = model_save_path + '_LeakyReLU_'
                train_label_name = 'leaky_relu_train'
                test_label_name = 'leaky_relu_test'
            elif i == 2:
                model_save_path_a = model_save_path + '_ReLU_'
                train_label_name = 'relu_train'
                test_label_name = 'relu_test'
            else:
                model_save_path_a = model_save_path + '_Unknown_'
                train_label_name = 'Unknown_train'
                test_label_name = 'Unknown_test'

            net = EEGNet(i)
            # net = DeepConvNet(i)
            net = net.to(device)
            # summary(net, [(1, 2, 750)])
            optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=w_decay)
            criterion = nn.CrossEntropyLoss().to(device)

            epoch_hist, acc_train_hist, acc_test_hist = train(net, optimizer, criterion, model_save_path_a)
            train_max[i] = np.max(acc_train_hist)
            test_max[i] = np.max(acc_test_hist)

            plt.plot(epoch_hist, acc_train_hist, label=train_label_name)
            plt.plot(epoch_hist, acc_test_hist, label=test_label_name)
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy(%)")
            plt.title("Activation function comparison(DeepConvNet)")
            plt.legend()

        print(train_max)
        print(test_max)
        plt.show()


demo(model_type='EEGNet', actvation_type=2, loop_number=1)
# demo_from_best_model(1, 256, 'backup/EEGNet_loop_0_ReLU_best.pkl')
# train_for_3_activation()

