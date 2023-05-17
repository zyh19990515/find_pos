# -*- coding:utf-8 -*-

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


def Myloader(path):
    return Image.open(path).convert('RGB')


# get a list of paths and labels.
def init_process(path, lens):
    data = []
    name = find_label(path)
    for i in range(lens[0], lens[1]):
        data.append([path % i, name])
        # print(data)

    return data


class MyDataset(Dataset):
    def __init__(self, data, transform, loader):
        self.data = data
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

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
            last = i - 1
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
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.3),
        # transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalization
    ])
    all_data_num = 2000
    base_path = '.\\data\\pic\\20230407-1\\'
    path1 = base_path + 'training_data\\1_\\%d.jpg'
    data1 = init_process(path1, [0, int(0.7*all_data_num)])
    path2 = base_path + 'training_data\\2_\\%d.jpg'
    data2 = init_process(path2, [0, int(0.7*all_data_num)])
    path3 = base_path + 'training_data\\3_\\%d.jpg'
    data3 = init_process(path3, [0, int(0.7*all_data_num)])
    path4 = base_path + 'training_data\\4_\\%d.jpg'
    data4 = init_process(path4, [0, int(0.7*all_data_num)])
    path5 = base_path + 'training_data\\5_\\%d.jpg'
    data5 = init_process(path5, [0, int(0.7*all_data_num)])
    path6 = base_path + 'training_data\\6_\\%d.jpg'
    data6 = init_process(path6, [0, int(0.7*all_data_num)])
    path7 = base_path + 'training_data\\7_\\%d.jpg'
    data7 = init_process(path7, [0, int(0.7*all_data_num)])
    path8 = base_path + 'training_data\\8_\\%d.jpg'
    data8 = init_process(path8, [0, int(0.7*all_data_num)])
    path9 = base_path + 'training_data\\9_\\%d.jpg'
    data9 = init_process(path9, [0, int(0.7*all_data_num)])
    path19 = base_path + 'training_data\\0_\\%d.jpg'
    data19 = init_process(path9, [0, int(0.7 * all_data_num)])

    path10 = base_path + 'testing_data\\1_\\%d.jpg'
    data10 = init_process(path10, [int(0.7*all_data_num) + 1, all_data_num-1])
    path11 = base_path + 'testing_data\\2_\\%d.jpg'
    data11 = init_process(path11, [int(0.7*all_data_num) + 1, all_data_num-1])
    path12 = base_path + 'testing_data\\3_\\%d.jpg'
    data12 = init_process(path12, [int(0.7*all_data_num) + 1, all_data_num-1])
    path13 = base_path + 'testing_data\\4_\\%d.jpg'
    data13 = init_process(path13, [int(0.7*all_data_num) + 1, all_data_num-1])
    path14 = base_path + 'testing_data\\5_\\%d.jpg'
    data14 = init_process(path14, [int(0.7*all_data_num) + 1, all_data_num-1])
    path15 = base_path + 'testing_data\\6_\\%d.jpg'
    data15 = init_process(path15, [int(0.7*all_data_num) + 1, all_data_num-1])
    path16 = base_path + 'testing_data\\7_\\%d.jpg'
    data16 = init_process(path16, [int(0.7*all_data_num) + 1, all_data_num-1])
    path17 = base_path + 'testing_data\\8_\\%d.jpg'
    data17 = init_process(path17, [int(0.7*all_data_num) + 1, all_data_num-1])
    path18 = base_path + 'testing_data\\9_\\%d.jpg'
    data18 = init_process(path18, [int(0.7*all_data_num) + 1, all_data_num-1])
    path20 = base_path + 'testing_data\\0_\\%d.jpg'
    data20 = init_process(path18, [int(0.7 * all_data_num) + 1, all_data_num - 1])
    data = data1 + data2 + data3 + data4 + data5 + data6 + data7 + data8 + data9 + data10 + data11 + data12 + data13 + data14 + data15 + data16 + data17 + data18 + data19 + data20 #18000
    # shuffle
    np.random.shuffle(data)
    # train, val, test = 900 + 200 + 300
    # train_data, val_data, test_data = data[:2400], data[2400:3400], data[3400:3900]
    sum = 10 * all_data_num
    train_data, val_data, test_data = data[:int(sum*0.6)], data[int(sum*0.6):int(sum*0.6)+int(sum*0.3)], data[int(sum*0.6)+int(sum*0.3):]
    # train_data, val_data, test_data = data[3900:4500], data[4500:4800], data[4800:]
    train_data = MyDataset(train_data, transform=transform, loader=Myloader)
    Dtr = DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=0)
    val_data = MyDataset(val_data, transform=transform, loader=Myloader)
    Val = DataLoader(dataset=val_data, batch_size=32, shuffle=True, num_workers=0)
    test_data = MyDataset(test_data, transform=transform, loader=Myloader)
    Dte = DataLoader(dataset=test_data, batch_size=32, shuffle=True, num_workers=0)

    return Dtr, Val, Dte
