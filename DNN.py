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
# from DataProcess import load_data
from dnn_data_process import load_data
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

class dnn(nn.Module):
    def __init__(self):
        super(dnn, self).__init__()
        self.net1 = nn.Linear(4, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.net2 = nn.Linear(16, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.net3 = nn.Linear(32, 10)
        self.net4 = nn.Linear(32, 32)
        self.drop = nn.Dropout1d(0.2)
        self.Lrelu = nn.LeakyReLU()


    def forward(self, x):
        x = self.net1(x)
        # x = self.drop(x)
        x = self.Lrelu(x)
        x = self.net2(x)
        x = self.Lrelu(x)
        # x = self.drop(x)
        # x = self.bn2(x)
        x = self.net4(x)

        x = self.Lrelu(x)
        x = self.net3(x)
        # x = self.drop(x)
        x = self.Lrelu(x)

        output = F.log_softmax(x, dim=1)

        return output

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
    epoch_num = 200
    best_model = None
    min_epochs = 5
    min_val_loss = 5
    model = dnn().to(device)
    # model.load_state_dict(torch.load("model/20230413/dnn3(y).pkl"), True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
    torch.save(best_model.state_dict(), "model/20230511/dnn4.pkl")
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
