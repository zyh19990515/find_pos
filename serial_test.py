import serial
import re

if __name__ == '__main__':
    s = serial.Serial('com5', 115200)
    Str = "3"
    while True:
        s.write(Str.encode('utf-8'))
        string = s.readline()
        print(string)
        # st1 = re.findall(r'\d+\.?\d*', string.decode('utf-8'))
        # print(st1)
