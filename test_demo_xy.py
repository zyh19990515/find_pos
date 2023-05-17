import serial
import torch
from DNN import dnn
import re
import cv2
import numpy as np

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



if __name__ == '__main__':

    s = serial.Serial('com6', 115200)

    flag = 0
    cnt = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_x = dnn().to(device)
    model_y = dnn().to(device)
    model_x.load_state_dict(torch.load("model/20230413/dnn3(x).pkl"), False)
    model_x.eval()
    model_y.load_state_dict(torch.load("model/20230413/dnn3(y).pkl"), False)
    model_y.eval()
    while True:
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
            flag = 1
        else:
            continue
        if (flag == 1):
                # try:
                #     st1 = re.findall(r'\d+\.?\d*', st)
                #     nums = [float(st1[0]), float(st1[1]), float(st1[2]), float(st1[3])]
                #
                #     print(st1)
                #     print(nums)
                #     img = getImg(nums)
                #
                #     image = Image.fromarray(img).convert('RGB')
                #
                #
                #     img_tensor = transform(image)
                #     # print(img_tensor.shape)
                #     outputs = model(img_tensor)
                #     print(outputs)
                #     predicted = torch.max(outputs.data, 1)[1].data
                #     print(predicted)
                # except:
                #     print("no data")
            st1 = re.findall(r'\d+\.?\d*', st)
            nums = [float(st1[0]), float(st1[1]), float(st1[2]), float(st1[3])]

                # print(st1)
                # print(nums)
            nums = init_process(nums).to(device)
                # print(img_tensor.shape)
            nums = torch.unsqueeze(nums, 0)
            outputs_x = model_x(nums)
            outputs_y = model_y(nums)

            predicted_x = torch.max(outputs_x.data, 1)[1].data
            pos_x = predicted_x.cpu().item()
            predicted_y = torch.max(outputs_y.data, 1)[1].data
            pos_y = predicted_y.cpu().item()
                # print("x: ", pos_x)
                # print("y: ", pos_y)
            send_str = "x:" + str(pos_x) + " y:" + str(pos_y)
            print(send_str)
                # s.write(send_str.encode('utf-8'))
                # s_send_str = ''
                # while True:
                #     try:
                #         char = str(s_send.read(), 'utf-8')
                #         s_send_str = s_send_str + char
                #     except:
                #         continue
                #     if (char == '\n'):

