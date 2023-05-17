import time
import numpy as np
import serial_collect
import cv2
import math
import xlrd
import pandas as pd
import re
import time
#设置8个档,0.4为一档


def data2degree(data, labal):
    if(data>3.3):
        data = 0.0
    if(labal == 1):
        # if(data<1.7):
        #     return 0
        tmp = (3.1 - 0.0)/8
        degree = int(data / tmp)
    elif(labal == 2):
        # if (data < 0.26):
        #     return 0
        tmp = (3.05 - 0.0) / 8
        degree = int(data / tmp)
    elif (labal == 3):
        # if (data < 0.68):
        #     return 0
        tmp = (3.09 - 0.0) / 8
        degree = int(data / tmp)
    elif (labal == 4):
        # if (data < 1.9):
        #     return 0
        tmp = (3.08 - 0.0) / 8
        degree = int(data / tmp)
    return degree

if __name__ == '__main__':
    num = str(5)
    rd_path = ".\\new_data\\20230407-3\\20230407_" + num + ".xls"
    book = pd.read_excel(rd_path)
    col1 = book["1"]
    col2 = book["2"]
    col3 = book["3"]
    col4 = book["4"]
    img = np.zeros((256, 256), dtype=np.uint16)
    print(img.shape)
    for i in range(0, 2000):
        # print(col1[i])
        degree1 = data2degree(col1[i], 1)
        degree2 = data2degree(col2[i], 2)
        degree3 = data2degree(col3[i], 3)
        degree4 = data2degree(col4[i], 4)
        img[0:127, 0:127] = degree1 * 32
        img[0:127, 128:255] = degree2 * 32
        img[128:255, 0:127] = degree3 * 32
        img[128:255, 128:255] = degree4 * 32
        # time.sleep(0.5)
        # cv2.imshow("1", img)
        path = "E:\\code\\python\\find_pos\\new_data\\202304067-3\\pic\\" + num + "_\\" + str(i) + ".jpg"
        path1 = "E:\\code\\python\\find_pos\\data\\pic\\20230407-3\\training_data\\" + num + "_\\" + str(i) + ".jpg"
        path2 = "E:\\code\\python\\find_pos\\data\\pic\\20230407-3\\testing_data\\" + num + "_\\" + str(i) + ".jpg"
        cv2.imwrite(path, img)
        cv2.imwrite(path1, img)
        cv2.imwrite(path2, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
