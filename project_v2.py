import serial
import torch
from DNN import dnn
import re
import cv2
import random
import numpy as np
import socket
'''
WiFi
'''

def init_process(nums):


    num1 = float(nums[0])
    if(num1>3.3):
        num1 = 0
    num2 = float(nums[1])
    if (num2 > 3.3):
        num2 = 0
    num3 = float(nums[2])
    if (num3 > 3.3):
        num3 = 0
    num4 = float(nums[3])
    if (num4 > 3.3):
        num4 = 0

        # nums = [float(col1[i]), float(col2[i]), float(col3[i]), float(col4[i])]
    nums = [num1, num2, num3, num4]
    nums = np.array(nums)
    nums = torch.tensor(nums, dtype=torch.float)

    return nums



if __name__ == '__main__':
    s1 = serial.Serial('com6', 115200)
    s2 = serial.Serial('com3', 115200)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_x = dnn().to(device)
    model_x.load_state_dict(torch.load(r"E:\code\python\find_pos\model\20230511\dnn4.pkl"), False)
    model_x.eval()
    flag = 0
    cnt = 1


    while True:
        st = ''
        while True:
            try:
                char = str(s1.read(), 'utf-8')
                st = st + char
            except:
                continue
            if (char == '\n'):
                break
        if (st[0] == 'd'):
            flag = 1
        else:
            continue
        if (flag == 1):
            st1 = re.findall(r'\d+\.?\d*', st)
            nums = [float(st1[0]), float(st1[1]), float(st1[2]), float(st1[3])]
            print(nums)
            nums = init_process(nums).to(device)
            nums = torch.unsqueeze(nums, 0)
            outputs_x = model_x(nums)
            predicted_x = torch.max(outputs_x.data, 1)[1].data
            pos_x = predicted_x.cpu().item()
            print(pos_x)
            send_str = str(pos_x) + "\n"
