import copy
import os
import random

import numpy as np
import torch
from DNN import dnn
from dnn_data_process import load_data

def test():
    Dtr, Val, Dte = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = dnn().to(device)
    model.load_state_dict(torch.load("model/20230511/dnn4.pkl"), False)
    model.eval()
    total = 0
    current = 0

    for (data, target) in Dtr:
        data, target = data.to(device), target.to(device)
        outputs = model(data)

        predicted = torch.max(outputs.data, 1)[1].data
        # print(predicted, target)
        total += target.size(0)
        for i in range(0, len(predicted)):
            if(predicted[i] != target[i]):
                print(predicted[i].cpu().item(), "  ", target[i].cpu().item())

        current += (predicted == target).sum()

    print('Accuracy:%d%%' % (100 * current / total))

if __name__ == '__main__':
    test()