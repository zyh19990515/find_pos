import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
from DataProcess import load_data
# from data_process import load_data
# from data_pro import load_data
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        #
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        #
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Linear(3 * 3 * 64, 64)
        self.fc2 = nn.Linear(64, 10)
        self.out = nn.Linear(10, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.size())
        x = x.view(x.shape[0], -1)
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        # x = self.sigmoid(self.out(x))
        x = F.log_softmax(x, dim=1)
        return x


def get_val_loss(model, Val):
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    val_loss = []
    for (data, target) in Val:
        data, target = Variable(data).to(device), Variable(target.long()).to(device)
        output = model(data)
        # print(output, target)
        loss = criterion(output, target)
        val_loss.append(loss.cpu().item())

    return np.mean(val_loss)


def train():
    Dtr, Val, Dte = load_data()
    print('train...')
    epoch_num = 30
    best_model = None
    min_epochs = 5
    min_val_loss = 5
    model = cnn().to(device)
    model.load_state_dict(torch.load("model/cnn9.pkl"), True)
    optimizer = optim.Adam(model.parameters(), lr=0.0008)
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = nn.BCELoss().to(device)
    test_acc_all = []
    train_loss_vector = []
    val_loss_vector = []
    for epoch in tqdm(range(epoch_num), ascii=True):
        train_loss = []
        for batch_idx, (data, target) in enumerate(Dtr, 0):
            data, target = Variable(data).to(device), Variable(target.long()).to(device)
            # target = target.view(target.shape[0], -1)
            # print(target)
            optimizer.zero_grad()
            output = model(data)
            # print(output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.cpu().item())
            # print(loss.cpu().item())
        # validation
        val_loss = get_val_loss(model, Val)
        # print("val_loss:", val_loss)
        test_acc = test(model, Dte)
        t = test_acc.cpu().item()

        test_acc_all.append(t)
        model.train()
        if epoch + 1 > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
        train_loss_vector.append(np.mean(train_loss))
        val_loss_vector.append(val_loss)
        tqdm.write('Epoch {:03d} train_loss {:.5f} val_loss {:.5f}'.format(epoch, np.mean(train_loss), val_loss))
    torch.save(best_model.state_dict(), "model/cnn11.pkl")
    plt.figure()
    plt.plot(test_acc_all)
    plt.figure()
    plt.plot(train_loss_vector)
    plt.plot(val_loss_vector)
    plt.show()



def test(model, Dte):
    # Dtr, Val, Dte = load_data()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = cnn().to(device)
    # model.load_state_dict(torch.load("model/cnn6.pkl"), False)
    model.eval()
    total = 0
    current = 0
    for (data, target) in Dte:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        predicted = torch.max(outputs.data, 1)[1].data
        total += target.size(0)
        current += (predicted == target).sum()

    print('Accuracy:%d%%' % (100 * current / total))
    return 100 * current / total


if __name__ == '__main__':
    train()
    # test()
