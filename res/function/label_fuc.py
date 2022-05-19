import numpy as np
import math
infinity = 100000
color_green = (0, 255, 0)
color_red = (0, 0, 255)
color_blue = (255, 0, 0)
def getCenter(x, y, img):
    shift = (__right__(x, y, img) - __left__(x, y, img))/2#需要平移的距離
    return (round(x + shift), y)
def getValue(x, y, m, img):
    width = __right__(x, y, img) + __left__(x, y, img) - 1
    if m == infinity:#直接計算
        value = width
    else:
        value = math.sin(math.atan(m)) * width
    return abs(round(value))
def getPoint(x, y, m, value, direct, whether_y):#輸入想位移的值以及斜率即可返還點座標
    tmpx = ((value * value) / (m * m + 1)) ** 0.5
    tmpy = m * tmpx
    if direct == "Left": #右手
        if whether_y == True and m > 0:
            return (round(x - tmpx), round(y + tmpy))
        return (round(x + tmpx), round(y - tmpy))
    else: #左手
        if whether_y == True and m < 0:
            return (round(x + tmpx), round(y - tmpy))
        return (round(x - tmpx), round(y + tmpy))
def find_first_x(a):
    alter = 0
    min = infinity
    min_i = []
    start = 0
    for i in range(len(a) - 1, -1, -1):
        if alter == 0 and a[i] == 0:
            alter = 1
        if alter == 1:
            if start == 0 and a[i] != 0:
                start = i
            if a[i] <= min and a[i] != 0:
                min = a[i]
    for i in range(start, -1, -1):
        if a[i] == min:
            min_i.append(i)
    if len(min_i) != 0:
        return min_i[int(len(min_i)/2)]
    else:
        print("please deal with error")
        alter = 0
        min = infinity
        min_i = []
        start = 0
        for i in range(len(a) - 1, -1, -1):
            if alter == 0 and abs(a[i-1] - a[i]) > len(a)/4:
                alter = 1
                continue
            if alter == 0 and a[i] == 0:
                alter = 1
            if alter == 1:
                if start == 0 and a[i] != 0:
                    start = i
                if a[i] <= min and a[i] != 0:
                    min = a[i]
        for i in range(start, -1, -1):
            if a[i] == min:
                min_i.append(i)
        if len(min_i) != 0:
            return min_i[int(len(min_i)/2)]
        return 0
def find_other_x(m, points, x, y, img):# 考慮到複雜度, 為了不每轉一次迴圈全部跑一遍確認有沒有點, 改成直接用碰撞到藍色的方式
    points = np.array(points)
    if m <= 1 and m >= -1:
        unit = abs(round(1/m))
        first_shift = abs(round((1/m)/2))
    else:
        unit = abs(round(m))
        first_shift = abs(round((m)/2))
    k = 0
    for i in range(1, img.shape[0]): #up
        tmp = np.where(points == [x + k, y - i])
        if len(tmp[0]) >= 2:
            for j in range(len(tmp[0])):
                if len(np.argwhere(tmp[0] == tmp[0][j])) == 2:
                    return tmp[0][np.argwhere(tmp[0] == tmp[0][j])[0][0]]
        if i - 1 == first_shift:
            if m < 0:
                k -= 1
            else:
                k += 1
        if (i - 1 - first_shift) % unit == 0:
            if m < 0:
                k -= 1
            else:
                k += 1
        tmp = np.where(points == [x + k, y - i])
        if len(tmp[0]) >= 2:
            for j in range(len(tmp[0])):
                if len(np.argwhere(tmp[0] == tmp[0][j])) == 2:
                    return tmp[0][np.argwhere(tmp[0] == tmp[0][j])[0][0]]
    return 0
def find_y_points(a, value):
    min1 = min2 = min3 = infinity
    alter1 = alter2 = alter3 = 0
    start1 = start2 = start3 = 0
    end1 = end2 = end3 = 0
    for i in range(1, len(a)):
        if end1 == 0:
            if abs(a[i] - a[i - 1]) > value and a[i] != 0 and a[i - 1] != 0 or alter1 == 1:
                alter1 = 1
                if start1 == 0:
                    start1 = i
                    continue
                if a[i] < min1:
                    min1 = a[i]
            
        if start1 != 0 and end1 == 0:
            if abs(a[i] - a[i - 1]) > value and a[i] != 0 and a[i - 1] != 0:
                end1 = i - 1
            continue
        if end1 != 0 and end2 == 0:
            if abs(a[i] - a[i - 1]) > value and a[i] != 0 and a[i - 1] != 0 or alter2 == 1:
                alter2 = 1
                if start2 == 0:
                    start2 = i
                    continue
                if a[i] < min2:
                    min2 = a[i]
            
        if start2 != 0 and end2 == 0:
            if abs(a[i] - a[i - 1]) > value and a[i] != 0 and a[i - 1] != 0:
                end2 = i - 1
                #print("end2:", i - 1)
            continue
        if end2 != 0 and end3 == 0:
            if abs(a[i] - a[i - 1]) > value and a[i] != 0 and a[i - 1] != 0 or alter3 == 1:
                #print("end3_1:", i,a[i])
                alter3 = 1
                if start3 == 0:
                    start3 = i
                    continue
                if a[i] < min3:
                    min3 = a[i]
            
        if start3 != 0 and end3 == 0:
            if abs(a[i] - a[i - 1]) > value and a[i] != 0 and a[i - 1] != 0:
                #print("end3_2:", i,a[i])
                end3 = i - 1
                break
    label1 = []
    label2 = []
    label3 = []
    for i in range(len(a)):
        if i >= start1 and i <= end1:
            if a[i] == min1:
                label1.append(i)
        if i >= start2 and i <= end2:
            if a[i] == min2:
                label2.append(i)
        if i >= start3 and i <= end3:
            if a[i] == min3:
                label3.append(i)
    #print("min1,2,3:",min1,min2,min3)
    # try:
    #     print("1, ", label1[0])
    # except:
    #     pass
    # try:
    #     print("2, ", label2[0])
    # except:
    #     pass
    # try:
    #     print("2, ", label3[0])
    # except:
    #     pass
    if len(label1) != 0 and len(label2) != 0 and len(label3) != 0:
        return (label1[int(len(label1)/2)], label2[int(len(label2)/2)], label3[int(len(label3)/2)])
    return (0, 0, 0)
def whetherPassFromYaxis(x, y, m, direct, value, img):#return(dis, first_x, first_y), dis等於0表示沒有
    ori = img[y][x]
    if m <= 1 and m >= -1:
        if m != 0:
            unit = abs(round(1/m))
            first_shift = abs(round((1/m)/2))
        else:
            unit = infinity
            first_shift = infinity
    else:
        unit = abs(round(m))
        first_shift = abs(round((m)/2))
    if direct == "Left": #Left是右手
        k = 0
        for i in range(1, img.shape[1]): #right
            if (i * i + k * k) ** 0.5 >= value:
                return (value, 0, 0)
            if (ori != img[y + k][x + i]).any():#記錄
                return ((i * i + k * k) ** 0.5, x + i, y + k)#紀錄該點根據x軸方向延伸後第一個碰到的邊界距離, 以及它的x, y
            if i - 1 == first_shift:
                if m < 0:
                    k += 1
                else:
                    k -= 1
            if (i - 1 - first_shift) % unit == 0:
                if m < 0:
                    k += 1
                else:
                    k -= 1
            if (ori != img[y + k][x + i]).any():#記錄if
                return ((i * i + k * k) ** 0.5, x + i, y + k)#紀錄該點根據x軸方向延伸後第一個碰到的邊界距離, 以及它的x, y
            if (i * i + k * k) ** 0.5 >= value:
                return (value, 0, 0)
    else:
        k = 0
        for i in range(1, img.shape[1]): #left
            if (i * i + k * k) ** 0.5 >= value:
                return (value, 0, 0)
            if (ori != img[y + k][x - i]).any():#記錄
                return ((i * i + k * k) ** 0.5, x - i, y + k)#紀錄該點根據x軸方向延伸後第一個碰到的邊界距離, 以及它的x, y
            if i - 1 == first_shift:
                if m < 0:
                    k -= 1
                else:
                    k += 1
            if (i - 1 - first_shift) % unit == 0:
                if m < 0:
                    k -= 1
                else:
                    k += 1
            if (ori != img[y + k][x - i]).any():#記錄else
                return ((i * i + k * k) ** 0.5, x - i, y + k)#紀錄該點根據x軸方向延伸後第一個碰到的邊界距離, 以及它的x, y
            if (i * i + k * k) ** 0.5 >= value:
                return (value, 0, 0)
    return (0,0,0)
def whetherPassFromXaxis(x, y, m, value,img):
    ori = img[y][x]
    if m <= 1 and m >= -1:
        if m != 0:
            unit = abs(round(1/m))
            first_shift = abs(round((1/m)/2))
        else:
            unit = infinity
            first_shift = infinity
    else:
        unit = abs(round(m))
        first_shift = abs(round((m)/2))
    k = 0
    first_touch_distance = 0
    for i in range(1, img.shape[0]): #down
        if (i * i + k * k) ** 0.5 >= value:
                return (0, 0, 0)
        if i - 1 == first_shift:
            if m < 0:
                k += 1
            else:
                k -= 1
        if (i - 1 - first_shift) % unit == 0:
            if m < 0:
                k += 1
            else:
                k -= 1
        if (ori != img[y + i][x + k]).any():
            first_touch_distance = (i * i + k * k) ** 0.5
            return (first_touch_distance, x + k, y + i)
        if (i * i + k * k) ** 0.5 >= value:
                return (0, 0, 0)
def getCollidePoint(x, y, m, direct, img):#斜率1/10的話, 表示x移動10格y往上一格, 實際上則需要往右五格
    ori = img[y][x]
    alter = 0
    if m <= 1 and m >= -1:
        alter = 1
        if m != 0:
            unit = abs(round(1/m))
            first_shift = abs(round((1/m)/2))
        else:
            unit = infinity
            first_shift = infinity
    else:
        unit = abs(round(m))
        first_shift = abs(round((m)/2))
    print("m ", m)
    print("first ",first_shift)
    print("unit ", unit)
    if direct == "Left": #Left是右手
        if alter == 1:
            k = 0
            for i in range(1, img.shape[1]): #right
                print(i)
                if i - 1 == first_shift:
                    if m < 0:
                        k += 1
                    else:
                        k -= 1
                if (i - 1 - first_shift) % unit == 0:
                    if m < 0:
                        k += 1
                    else:
                        k -= 1
                if (ori != img[y + k][x + i]).any():
                    return (x + i, y + k)
    else:
        if alter == 1:
            k = 0
            for i in range(1, img.shape[1]): #left
                if i - 1 == first_shift:
                    if m < 0:
                        k -= 1
                    else:
                        k += 1
                if (i - 1 - first_shift) % unit == 0:
                    if m < 0:
                        k -= 1
                    else:
                        k += 1
                if (ori != img[y + k][x - i]).any():
                    return (x - i, y + k)
    return (0,0)
        
def __left__(x, y, img):
    ori = img[y][x]
    for i in range(1, img.shape[1]): #left
        if (ori != img[y][x-i]).any():
            return i
    return 0

def __right__(x, y, img):
    ori = img[y][x]
    for i in range(1, img.shape[1]): #right
        if (ori != img[y][x+i]).any():
            return i
    return 0