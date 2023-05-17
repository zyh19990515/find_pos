import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlwt
if __name__ == '__main__':
    book = pd.read_excel(".\\new_data\\Calibration\\20230407_1\\20230407_1.xls")
    data1 = book["1"]
    data2 = book["2"]
    data3 = book["3"]
    data4 = book["4"]

    for i in range(0, 2000):
        if(data1[i] > 3.3):
            data1[i] = 0
        if (data2[i] > 3.3):
            data2[i] = 0
        if (data3[i] > 3.3):
            data3[i] = 0
        if (data4[i] > 3.3):
            data4[i] = 0
    data1_max = data1.max()
    data2_max = data2.max()
    data3_max = data3.max()
    data4_max = data4.max()

    data1_min = data1.min()
    data2_min = data2.min()
    data3_min = data3.min()
    data4_min = data4.min()

    ma = 0
    mi = 999
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.tight_layout(h_pad=2)
    plt.plot(data1)
    plt.title("max:" + str(data1_max) + " min:" + str(data1_min))
    plt.subplot(2, 2, 2)
    plt.tight_layout(h_pad=2)
    plt.plot(data2)
    plt.title("max:" + str(data2_max) + " min:" + str(data2_min))
    plt.subplot(2, 2, 3)
    plt.tight_layout(h_pad=2)
    plt.plot(data3)
    plt.title("max:" + str(data3_max) + " min:" + str(data3_min))
    plt.subplot(2, 2, 4)
    plt.tight_layout(h_pad=2)
    plt.plot(data4)
    plt.title("max:" + str(data4_max) + " min:" + str(data4_min))
    plt.show()

    book = xlwt.Workbook(encoding='utf-8')
    sheet = book.add_sheet('1')
    sheet.write(1, 0, 'max')
    sheet.write(2, 0, 'min')
    sheet.write(0, 1, '1')
    sheet.write(0, 2, '2')
    sheet.write(0, 3, '3')
    sheet.write(0, 4, '4')

    sheet.write(1, 1, str(data1_max))
    sheet.write(2, 1, str(data1_min))
    sheet.write(1, 2, str(data2_max))
    sheet.write(2, 2, str(data2_min))
    sheet.write(1, 3, str(data3_max))
    sheet.write(2, 3, str(data3_min))
    sheet.write(1, 4, str(data4_max))
    sheet.write(2, 4, str(data4_min))

    book.save(".\\new_data\\Calibration\\20230407_1\\done_20230407_1.xls")


    # for i in range(0, 2000):
    #     if(i > 3.3):
    #         i = 0
    #     if(i<mi):
    #         mi = i
    #     if(i>ma):
    #         ma = i
    # print("max:", ma)
    # print("min:", mi)
    # plt.figure()
    # plt.plot(data)
    # plt.show()
    # df = pd.DataFrame({'1' : data})
    # df.to_excel(".\\new_data\\Calibration\\done_20230405.xls")
