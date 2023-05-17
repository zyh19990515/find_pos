import pandas as pd
import xlwt
import matplotlib.pyplot as plt

if __name__ == '__main__':
    rd_path = ".\\new_data\\20230504\\y\\20230504_4.xls"
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
        if (float(col1[i]) > 3.3 or float(col1[i])<0.1):
            col1[i] = 0.0
        if (float(col2[i]) > 3.3 or float(col2[i])<0.1):
            col2[i] = 0.0
        if (float(col3[i]) > 3.3 or float(col3[i])<0.1):
            col3[i] = 0.0
        if (float(col4[i]) > 3.3 or float(col4[i])<0.1):
            col4[i] = 0.0
    book.to_excel(".\\new_data\\20230504\\data\\y\\4.xls")

    # plt.figure()
    # plt.subplot(4, 1, 1)
    # plt.plot(data1, 'r-')
    # plt.subplot(4, 1, 2)
    # plt.plot(data2, 'g-')
    # plt.subplot(4, 1, 3)
    # plt.plot(data3, 'b-')
    # plt.subplot(4, 1, 4)
    # plt.plot(data4, 'y-')
    # plt.show()