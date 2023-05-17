import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



if __name__ == '__main__':
    rd_path = ".\\new_data\\20230412-2\\all.xls"
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
    for i in range(0, col1.shape[0]):
        # if (col1[i] > 80):
        #     col1[i] = 0
        data1.append(float(col1[i]))
        # if(col2[i]>80):
        #     col2[i] = 0
        data2.append(float(col2[i]))
        # if (col3[i] > 80):
        #     col3[i] = 0
        data3.append(float(col3[i]))
        # if (col4[i] > 80):
        #     col4[i] = 0
        data4.append(float(col4[i]))
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(data1, 'r-')
    plt.subplot(4, 1, 2)
    plt.plot(data2, 'g-')
    plt.subplot(4, 1, 3)
    plt.plot(data3, 'b-')
    plt.subplot(4, 1, 4)
    plt.plot(data4, 'y-')
    plt.show()
