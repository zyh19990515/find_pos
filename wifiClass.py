import pywifi
from pywifi import const
import socket
import time
class wifi_control():
    def __init__(self):
        super(wifi_control, self).__init__()
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

if __name__ == '__main__':
    wifi = wifi_control()
    # wifi.connect()
    wifi.control()
    while True:
        wifi.senddata(1)
