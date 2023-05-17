import serial
import torch
from DNN import dnn
import re
import cv2
import numpy as np
def generateImg(pos):
    img = np.zeros((300, 300), dtype=np.uint8)
    if (pos == 1):
        img[0:99, 0:99] = 255
    elif (pos == 2):
        img[0:99, 100:199] = 255
    elif (pos == 3):
        img[0:99, 200:299] = 255
    elif (pos == 4):
        img[100:199, 0:99] = 255
    elif (pos == 5):
        img[100:199, 100:199] = 255
    elif (pos == 6):
        img[100:199, 200:299] = 255
    elif (pos == 7):
        img[200:299, 0:99] = 255
    elif (pos == 8):
        img[200:299, 100:199] = 255
    elif (pos == 9):
        img[200:299, 200:299] = 255
    return img

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
    if __name__ == '__main__':
        s = serial.Serial('com6', 115200)
        flag = 0
        cnt = 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = dnn().to(device)
        model.load_state_dict(torch.load(r"E:\code\python\find_pos\model\20230511\dnn4.pkl"), False)
        model.eval()
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

                print(st1)
                print(nums)
                nums = init_process(nums).to(device)
                # print(img_tensor.shape)
                nums = torch.unsqueeze(nums, 0)
                outputs = model(nums)

                predicted = torch.max(outputs.data, 1)[1].data
                pos = predicted.cpu().item()
                print(pos)
                img = generateImg(pos)
                cv2.imshow("pos1", img)
                k = cv2.waitKey(1)
                if (k == 27):
                    cv2.destroyAllWindows()
                    break