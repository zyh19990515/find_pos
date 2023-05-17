import serial
import torch
from DNN import dnn
import re
import cv2
import random
import numpy as np
from wifiClass import wifi_control
import socket
import threading
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
class WiFi_project():
    def __init__(self):
        super(WiFi_project, self).__init__()
        self.s_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    def control(self):
        fail_count = 0

        while (True):
            try:
                print("开始连接到服务器：\n")
                self.s_socket.connect(('192.168.137.193', 100))
                break
            except socket.error:
                fail_count = fail_count + 1
                print("连接服务器失败")
                if fail_count == 100:
                    return
    def senddata(self, send_str):
        #给服务器发送0，服务器接收后发送数据
        print("send successful")
        self.s_socket.send(send_str)

class Serial_project():
    def __init__(self, com, baud):
        self.com = com
        self.baud = baud
        self.s = serial.Serial(self.com, self.baud)
        self.flag = 0
        self.nums = []
    def revData(self):
        st = ''
        while True:
            try:
                char = str(s.read(), 'utf-8')
                st = st + char
            except:
                continue
            if (char == '\n'):
                break
        if (st[0] == 'd'):
            self.flag = 1
        if (self.flag == 1):
            st1 = re.findall(r'\d+\.?\d*', st)
            self.nums = [float(st1[0]), float(st1[1]), float(st1[2]), float(st1[3])]


if __name__ == '__main__':
    s = Serial_project('com6', 115200)
    w = WiFi_project()
    w.control()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_x = dnn().to(device)
    model_x.load_state_dict(torch.load(r"E:\code\python\find_pos\model\20230511\dnn1.pkl"), False)
    model_x.eval()
    flag = 0
    cnt = 1
    s_t = threading.Thread(target=s.revData)


    while True:
        s.revData()
        nums = init_process(s.nums).to(device)
            # print(img_tensor.shape)
        nums = torch.unsqueeze(nums, 0)
        outputs_x = model_x(nums)

        predicted_x = torch.max(outputs_x.data, 1)[1].data
        pos_x = predicted_x.cpu().item()

        send_str = str(pos_x) + "\n"
        w.senddata(send_str.encode("utf-8"))

