import xlwt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

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
    path = ".\\new_data\\20230412-1\\"
    labal_num = len(os.listdir(path))
    # labal = 1
    # channel = str(labal)
    data1 = []
    xbook = xlwt.Workbook(encoding='utf-8')
    sheet = xbook.add_sheet('points')
    sheet.write(0, 0, '1')
    sheet.write(0, 1, '2')
    sheet.write(0, 2, '3')
    sheet.write(0, 3, '4')

    for i in range(0, 4):
        labal = i + 1
        channel = str(labal)
        cnt = 1
        for i in os.listdir(path):
            tmp = path + i
            if(tmp[-1] == 'g'):
                continue
            book = pd.read_excel(tmp)
            # print(tmp)
            col1 = book[channel]
            kf1 = KF(1, 1, 1, 1, 0.1, 1, 0)

            for i in col1:
                num = float(i)
                if(num>3.3):
                    num = 0.0
                if(num<1.50):
                    num = 0.0
                # print(tmp[len(tmp)-5])
                if(labal == 3 and (tmp[len(tmp)-5] == 5 or tmp[len(tmp)-5] == 4)):
                    num += 0.3
                num = kf1.estimate(num)
                data1.append(num)
                sheet.write(cnt, labal-1, str(num))
                cnt += 1
    xbook.save(".\\new_data\\20230412-1\\all.xls")
    # plt.figure()
    # plt.plot(data1)
    # # plt.savefig(".\\new_data\\20230412-1\\" + channel + ".png")
    # plt.show()
