import pandas as pd
import xlwt
import matplotlib.pyplot as plt

if __name__ == '__main__':
    rd_path = ".\\new_data\\20230413-1\\x\\202304130.xls"
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
        col4[i] = 0.0
    book.to_excel(".\\new_data\\20230413-1\\x\\20230413_0.xls")