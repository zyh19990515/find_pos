import os
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def Myloader(path):
    return Image.open(path).convert('RGB')

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
    labal_num = 4
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.3),
        # transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalization
    ])

    path = ".\\data\\pic\\20230407-2\\training_data\\"
    train_Data = []
    test_Data = []
    all_data_num = 2000

    for i in os.listdir(path):
        tmp_path = path + i + "\\%d.jpg"
        train_data = init_process(tmp_path, [0, all_data_num])
        train_Data.append(train_data)
    data = []
    for i in train_Data:
        data += i

    np.random.shuffle(data)
    sum = labal_num * all_data_num

    train_data, val_data, test_data = data[:int(sum * 0.6)], data[int(sum * 0.6):int(sum * 0.6) + int(sum * 0.3)], data[
                                                                                                                   int(
                                                                                                                       sum * 0.6) + int(
                                                                                                                       sum * 0.3):]
    # train_data, val_data, test_data = data[3900:4500], data[4500:4800], data[4800:]
    train_data = MyDataset(train_data, transform=transform, loader=Myloader)
    Dtr = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)
    val_data = MyDataset(val_data, transform=transform, loader=Myloader)
    Val = DataLoader(dataset=val_data, batch_size=64, shuffle=True, num_workers=0)
    test_data = MyDataset(test_data, transform=transform, loader=Myloader)
    Dte = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0)

    return Dtr, Val, Dte


