import re
import serial
import numpy as np
import time
import xlwt

# 	1:	0.95	2:	0.00	3:	0.78	4:	1.70

def ToOmega(nums):
    st = []
    for i in nums:
        if(i < 0.1 or i > 3.3):
            st.append(str(0))
        else:

            i = 100*(3.28-float(i))/float(i)
            if(i > 200.0):
                i = 0.0
            st.append(str(i))
    return st


if __name__ == '__main__':
    s = serial.Serial('com6', 115200)
    book = xlwt.Workbook(encoding='utf-8')
    sheet = book.add_sheet('points')
    sheet.write(0, 0, '1')
    sheet.write(0, 1, '2')
    sheet.write(0, 2, '3')
    sheet.write(0, 3, '4')
    # sheet.write(0, 4, '5')
    flag = 0
    cnt = 1
    break_num = 5000
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
        st1 = re.findall(r'\d+\.?\d*', st)

        print(str(cnt)+':')
        print(st1)
        if(flag == 1):
            # nums = [float(st1[0]), float(st1[1]) + 0.128, float(st1[2]) + 0.0773, float(st1[3]) - 0.1439]
            nums = [float(st1[0]), float(st1[1]), float(st1[2]), float(st1[3])]
            print(nums)
            # st1 = ToOmega(nums)
            # sheet.write(cnt, 0, st1[0])
            # sheet.write(cnt, 1, st1[1])
            # sheet.write(cnt, 2, st1[2])
            # sheet.write(cnt, 3, st1[3])
            sheet.write(cnt, 0, str(nums[0]))
            sheet.write(cnt, 1, str(nums[1]))
            sheet.write(cnt, 2, str(nums[2]))
            sheet.write(cnt, 3, str(nums[3]))
            # sheet.write(cnt, 4, str(nums[4]))
            cnt += 1
        if(cnt == break_num + 1):
            break
    book.save(".\\new_data\\20230511\\20230511_9.xls")
