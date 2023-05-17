import os
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import xlwt
import pandas as pd
from DNN import dnn
import seaborn as sns
import matplotlib.pyplot as plt

def Myloader(nums):
    nums = np.array(nums)
    nums = torch.tensor(nums, dtype=torch.float)
    return nums

def init_process(path, lens):
    data = []
    name = find_label(path)
    rd_path = path
    book = pd.read_excel(rd_path)
    col1 = book["1"]
    col2 = book["2"]
    col3 = book["3"]
    col4 = book["4"]

    for i in range(lens[0], lens[1] + 1):
        num1 = float(col1[i])
        if(num1>3.3):
            num1 = 0
        num2 = float(col2[i])
        if (num2 > 3.3):
            num2 = 0
        num3 = float(col3[i])
        if (num3 > 3.3):
            num3 = 0
        num4 = float(col4[i])
        if (num4 > 3.3):
            num4 = 0

        # nums = [float(col1[i]), float(col2[i]), float(col3[i]), float(col4[i])]
        nums = [num1, num2, num3, num4]

        data.append([nums, name])
        # print(data)

    return data

class MyDataset(Dataset):
    def __init__(self, data, loader):
        self.data = data
        self.loader = loader

    def __getitem__(self, item):
        nums, label = self.data[item]
        nums = self.loader(nums)
        return nums, label

    def __len__(self):
        return len(self.data)

def find_label(str):
    """
    Find image tags based on file paths.

    :param str: file path
    :return: image label
    """
    first, last = 0, 0
    for i in range(len(str) - 1, -1, -1):
        if str[i] == '_':
            last = i + 1
            break
        # if (str[i] == 'c' or str[i] == 'd') and str[i - 1] == '/':
        #     first = i
        #     break

    name = str[last]
    if name == '1':
        return 1
    elif name == '2':
        return 2
    elif name == '3':
        return 3
    elif name == '4':
        return 4
    elif name == '5':
        return 5
    elif name == '6':
        return 6
    elif name == '7':
        return 7
    elif name == '8':
        return 8
    elif name == '9':
        return 9
    elif name == '0':
        return 0

def load_data():
    print('data processing')

    all_data_num = 500
    batch_size = 1
    path = ".\\new_data\\20230511\\"
    labal_num = len(os.listdir(path))
    data = []
    for i in os.listdir(path):
        tmp = path + i
        print(tmp)
        train_data = init_process(tmp, [0, all_data_num-1])
        data += train_data

    np.random.shuffle(data)
    sum = labal_num * all_data_num
    test_data = data

    test_data = MyDataset(test_data, loader=Myloader)
    Dtr = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=0)
    return Dtr

if __name__ == '__main__':
    test = load_data()
    acc_cnt = 0
    total = 0
    matrix = np.zeros((9, 9), dtype=np.int32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_x = dnn().to(device)
    model_x.load_state_dict(torch.load(r"E:\code\python\find_pos\model\20230511\dnn4.pkl"), False)
    model_x.eval()
    book = xlwt.Workbook(encoding='utf-8')
    sheet = book.add_sheet('confusion matrix')
    for (data, target) in test:
        data, target = data.to(device), target.to(device)
        outputs = model_x(data)

        predicted = torch.max(outputs.data, 1)[1].data
        # print(predicted, target)
        predicte = predicted.cpu().item()
        if(predicte == target):
            acc_cnt+=1
        matrix[target-1][predicte-1]+=1
        total+=1
    print(matrix)
    over_acc = float(acc_cnt/total)
    Datafame = pd.DataFrame(matrix, range(1,10), range(1,10))
    sns.heatmap(Datafame, annot=True, fmt='g')
    plt.xlabel('predicted labal')
    plt.ylabel('labal')
    plt.title('overall accuracy:%.3f '%over_acc)
    plt.show()
    # for i in range(5):
    #     for j in range(5):
    #         sheet.write(i, j, str(matrix[i][j]))
    # book.save(".\\new_data\\20230413-1\\test\\x\\confusion_matrix.xls")




