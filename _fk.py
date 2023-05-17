import numpy as np
import pandas as pd
# from kalmanfilter import KalmanFilter
import matplotlib.pyplot as plt

class KF:
    def __init__(self, my_A, my_C, my_P_, my_P, my_Q, my_R, my_x_):
        self.A = my_A
        self.C = my_C
        self.P_ = my_P_
        self.P = my_P
        self.Q = my_Q
        self.R = my_R
        self.x_ = my_x_
    def estimate(self, zk):
        self.P_ = self.A * self.P * self.A + self.Q
        self.K = (self.P_ * self.C) / ((self.C * self.P_ * self.C) + self.R)
        x = self.x_ + self.K * (zk - self.C * self.x_)
        self.P = (1 - self.K * self.C) * self.P_
        self.x_ = x
        return x
if __name__ == '__main__':
    rd_path = ".\\new_data\\20230504\\data\\x\\all.xls"
    book = pd.read_excel(rd_path)
    col1 = book["1"]
    col2 = book["2"]
    col3 = book["3"]
    col4 = book["4"]
    lens = col1.shape
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    thread_num = 3.3
    for i in range(0, col1.shape[0]):
        # if (col1[i] > thread_num):
        #     col1[i] = 0
        # data1.append(float(col1[i]))
        # if (col2[i] > thread_num):
        #     col2[i] = 0
        # data2.append(float(col2[i]))
        # if (col3[i] > thread_num):
        #     col3[i] = 0
        # data3.append(float(col3[i]))
        # if (col4[i] > thread_num):
        #     col4[i] = 0
        # data4.append(float(col4[i]))
        # if (col1[i] > thread_num):
        #     col1[i] = 0
        # data1.append(float(col1[i]))
        # if (col2[i] > thread_num):
        #     col2[i] = 0
        # data2.append(float(col2[i]))
        # if (col3[i] > thread_num):
        #     col3[i] = 0
        # data3.append(float(col3[i]))
        # if (col4[i] > thread_num):
        #     col4[i] = 0
        # data4.append(float(col4[i]))
        if (float(col1[i]) > 3.3 or float(col1[i])<0.1):
            col1[i] = 0
        if (float(col2[i]) > 3.3 or float(col2[i])<0.1):
            col2[i] = 0
        if (float(col3[i]) > 3.3 or float(col3[i])<0.1):
            col3[i] = 0
        if (float(col4[i]) > 3.3 or float(col4[i])<0.1):
            col4[i] = 0
        data1.append(float(col1[i]))
        data2.append(float(col2[i]))
        data3.append(float(col3[i]))
        data4.append(float(col4[i]))
    _data1 = []
    _data2 = []
    _data3 = []
    _data4 = []
    kf1 = KF(1, 1, 1, 1, 0.1, 1, 0)
    kf2 = KF(1, 1, 1, 1, 0.1, 1, 0)
    kf3 = KF(1, 1, 1, 1, 0.1, 1, 0)
    kf4 = KF(1, 1, 1, 1, 0.1, 1, 0)
    sum = 0
    for i in  range(0, len(data1)):
        _data1.append(kf1.estimate(data1[i]))
        _data2.append(kf2.estimate(data2[i]))
        _data3.append(kf3.estimate(data3[i]))
        _data4.append(kf4.estimate(data4[i]))




    # plt.figure(1)
    # plt.plot(data1,  'r-')
    # plt.plot(data2, 'g-')
    # plt.plot(data3, 'b-')
    # plt.plot(data4, 'y-')
    plt.figure(1)
    plt.subplot(4, 1, 1)
    plt.plot(_data1, 'r-')
    plt.subplot(4, 1, 2)
    plt.plot(_data2, 'g-')
    plt.subplot(4, 1, 3)
    plt.plot(_data3, 'b-')
    plt.subplot(4, 1, 4)
    plt.plot(_data4, 'y-')
    # plt.savefig(".\\new_data\\Calibration\\20230412-1\\4.png")
    plt.show()