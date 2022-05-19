import cv2, os
import mediapipe as mp
from res.function.label_fuc import getCenter, getValue, getPoint, whetherPassFromYaxis, whetherPassFromXaxis, find_y_points, find_first_x, find_other_x
import matplotlib.pyplot as plt
#define
infinity = 10000
color_green = (0, 255, 0)
color_red = (0, 0, 255)
color_blue = (255, 0, 0)
K = 3 #Y軸是X軸的三倍
whether_point = 0

def detect_label(img, const_rate, whether_show, grid_number, width_rate, alter_orix, alter_oriy, low_bound, upper_bound):#img is cv2 format
    xaxis_grid = grid_number
    yaxis_grid = K * xaxis_grid
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.5, max_num_hands = 1)
    mpDraw = mp.solutions.drawing_utils
    handLmsStyle = mpDraw.DrawingSpec(color = color_blue, thickness=3)
    handConStyle = mpDraw.DrawingSpec(color = color_green, thickness=5)


    
    
    #img = cv2.imread('test/photo/6.PNG')
    result = hands.process(img)
    
    #二值化描邊
    img = cv2.Canny(img, low_bound, upper_bound)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ####

    if result.multi_hand_landmarks:
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        originx = 0 #腕關節
        originy = 0
        hand9x = 0
        hand9y = 0
        for handLms in result.multi_hand_landmarks: 
            for i, lm in enumerate(handLms.landmark):
                xPos = round(lm.x * imgWidth)
                yPos = round(lm.y * imgHeight)
                if i == 0:
                    originx = xPos
                    originy = yPos
                if i == 9:
                    hand9x = xPos
                    hand9y = yPos
    dis = ((hand9x - originx)**2 + (hand9y - originy)**2)**0.5 #找斜率用的點距離
    originx += 0     #如果檢測的關節沒有很準***********************************************************************
    originy += 10
    m_x, m_y = getCenter(originx, round(originy + dis*0.75), img) #找斜率用的x和y
    #m_x, m_y = getCenter(originx, round(originy + (dis * 0.8)), img) #找斜率用的x和y
    if m_x == originx:
        print("infinity")
        ydir_m = infinity
    else:
        ydir_m = -1 * (originy - m_y) / (originx - m_x)
    if  ydir_m == infinity:
        xdir_m = 0
    else:
        xdir_m = -1 / ydir_m

    xdirValue = getValue(m_x, m_y, ydir_m, img)/width_rate #如果手臂形狀很特別***********************************************************************
    #視少數奇怪的人之情況自行更動值
    ydirValue = round(K * xdirValue)
    xdirValue = round(xdirValue)
    xdirEnd_x, xdirEnd_y = getPoint(originx, originy, xdir_m,  xdirValue, result.multi_handedness[0].classification[0].label, False)
    ydirEnd_x, ydirEnd_y = getPoint(originx, originy, ydir_m,  ydirValue, result.multi_handedness[0].classification[0].label, True)
    xdir_point = [[0 for _ in range(2)] for _ in range(xdirValue)]#每個點有x和y
    ydir_point = [[0 for _ in range(2)] for _ in range(ydirValue)]#每個點有x和y
    for i in range(xdirValue):
        xdir_point[i][0], xdir_point[i][1] = getPoint(originx, originy, xdir_m, i, result.multi_handedness[0].classification[0].label, False)
    for i in range(ydirValue):
        ydir_point[i][0], ydir_point[i][1] = getPoint(originx, originy, ydir_m, i, result.multi_handedness[0].classification[0].label, True)
    whethery = [0] * ydirValue #紀錄每個點所碰到的最近距離, 0表示沒有成功通過兩個點以上
    ycollid_point = [[0 for _ in range(2)] for _ in range(ydirValue)]#每個點有x和y
    for i in range(ydirValue):
        whethery[i], ycollid_point[i][0],  ycollid_point[i][1] = whetherPassFromYaxis(ydir_point[i][0], ydir_point[i][1], xdir_m, result.multi_handedness[0].classification[0].label, xdirValue, img)
        #print(whethery[i])
    ylabels = [0] * 3
    ylabels[0], ylabels[1], ylabels[2] = find_y_points(whethery, xdirValue/6)#10是基於人的手臂粗度的一半不可能貼超過10張
    #print("y:", ylabels)
    xcollid_point = [[0 for _ in range(2)] for _ in range(xdirValue)]#每個點有x和y
    whetherx = [0] * xdirValue #紀錄每個點所碰到的最近距離, 0表示沒有成功通過兩個點以上
    for i in range(xdirValue):
        whetherx[i], xcollid_point[i][0], xcollid_point[i][1] = whetherPassFromXaxis(xdir_point[i][0], xdir_point[i][1], ydir_m, ydirValue, img)
        #print(i, whetherx[i])
    xlabels = [0] * 3
    xlabels[0] = find_first_x(whetherx)
    #print("x:", xlabels)
    shift_x = xcollid_point[xlabels[0]][0] - ycollid_point[ylabels[0]][0]
    shift_y = xcollid_point[xlabels[0]][1] - ycollid_point[ylabels[0]][1]
    xlabels[1] = find_other_x(ydir_m, xdir_point, ycollid_point[ylabels[1]][0] + shift_x, ycollid_point[ylabels[1]][1] + shift_y, img)
    xlabels[2] = find_other_x(ydir_m, xdir_point, ycollid_point[ylabels[2]][0] + shift_x, ycollid_point[ylabels[2]][1] + shift_y, img)
    ####################################print################################################
    cv2.circle(img, (originx, originy), int(xdirValue/10), color_red, -1)   
    cv2.circle(img, (xdirEnd_x, xdirEnd_y), int(xdirValue/10), color_red, -1)
    cv2.line(img, (originx, originy), (xdirEnd_x, xdirEnd_y), color_red, 5)
    cv2.circle(img, (ydirEnd_x, ydirEnd_y), int(xdirValue/10), color_red, -1)
    cv2.line(img, (originx, originy), (ydirEnd_x, ydirEnd_y), color_red, 5)
    if whether_show == True:
        cv2.circle(img, (m_x, m_y), 10, color_blue, -1)          #用這條來顯示當前選擇的手臂粗度位置***********************
    if result.multi_handedness[0].classification[0].label == "Left":# hand type
        print("hand type: right")
    else: 
        print("hand type: left")
    for i in range(xdirValue):
        cv2.circle(img, (xdir_point[i][0], xdir_point[i][1]), 1, color_blue, -1)
    for i in range(ydirValue):
        cv2.circle(img, (ydir_point[i][0], ydir_point[i][1]), 1, color_blue, -1)
    #只是為了print出網格
    xgrid_point = [[0 for _ in range(2)] for _ in range(xaxis_grid)]#每個點有x和y
    ygrid_point = [[0 for _ in range(2)] for _ in range(yaxis_grid)]#每個點有x和y
    for i in range(xaxis_grid):
        xgrid_point[i][0], xgrid_point[i][1] = getPoint(originx, originy, xdir_m, (i/xaxis_grid  * xdirValue), result.multi_handedness[0].classification[0].label, False)
    for i in range(yaxis_grid):
        ygrid_point[i][0], ygrid_point[i][1] = getPoint(originx, originy, ydir_m, (i/yaxis_grid * ydirValue), result.multi_handedness[0].classification[0].label, True)
    another_xdir_point = [[0 for _ in range(2)] for _ in range(xaxis_grid)]#每個點有x和y
    another_ydir_point = [[0 for _ in range(2)] for _ in range(yaxis_grid)]#每個點有x和y
    for i in range(xaxis_grid):
        another_xdir_point[i][0], another_xdir_point[i][1] = getPoint(xgrid_point[i][0], xgrid_point[i][1], ydir_m, ydirValue, result.multi_handedness[0].classification[0].label, True)
    for i in range(yaxis_grid):
        another_ydir_point[i][0], another_ydir_point[i][1] = getPoint(ygrid_point[i][0], ygrid_point[i][1], xdir_m, xdirValue, result.multi_handedness[0].classification[0].label, False)
    for i in range(xaxis_grid):
        cv2.line(img, (xgrid_point[i][0], xgrid_point[i][1]), (another_xdir_point[i][0], another_xdir_point[i][1]), color_red, 1)
    for i in range(yaxis_grid):
        cv2.line(img, (ygrid_point[i][0], ygrid_point[i][1]), (another_ydir_point[i][0], another_ydir_point[i][1]), color_red, 1)
    #只是為了print出網格end

    for i in range(3):
        if xlabels[i] != 0:
            xlabels[i] = int(xlabels[i] / (xdirValue / xaxis_grid)) + 1
        if ylabels[i] != 0:
            ylabels[i] = int(ylabels[i] / (ydirValue / yaxis_grid)) + 1
    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)   
    return ('((' + str(xlabels[0]) + ',' + str(ylabels[0]) + '),(' + str(xlabels[1]) + ',' + str(ylabels[1]) + '),(' + str(xlabels[2]) + ',' + str(ylabels[2]) + '))', img)