# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:09:12 2020

CNN Visualization

@author: DELL
"""

import cv2
import numpy as np
import sys
#下面这两个list可以随意更改
#元素类型是（上方文字描述，通道数，feature map width，feature map height)如果w和h均是1代表是全连接层。作者qq@603997262
CNN_list=[
    ("input",1,28,28),
    ("hidden",28*28,1,1),
    ("hidden",64*64,1,1),
    ("feature",1,64,64),
    ("feature",4,64,64),
    ("feature",4,32,32),
    ("feature",8,32,32),
    ("feature",8,16,16),
    ("feature",8,16,16),
    ("feature",8,8,8),
    ("hidden",8*8*8,1,1),
    ("output",10,1,1),
 
]
#元素类型是（文字描述,卷积核宽度，卷积核高度）
operation_list=[
    ("flatten", 0, 0),#卷积核高度和宽度都是0代表flatten
    ("fully connect",-1,-1),
    ("stack", 0, 0),
    ("conv",3,3),
    ("maxpooling", 2, 2),
    ("conv", 3, 3),
    ("maxpooling", 2, 2),
    ("conv", 3, 3),
    ("maxpooling", 2, 2),
    ("flatten",0,0),#卷积核高度和宽度都是0代表flatten
    ("fully connect", -1, -1),#卷积核高度和宽度都是-1代表 全连接层
]#注意，operation_list的长度应该比CNN_list少1
#这是中间绘图区（包括说明，神经网络，指示器）的宽度
 
#以下参数如果读懂可以更改
OVERALL_SCALE=0.8#控制整个图像和其中元素的缩放，1为原始尺寸
FONT_FACE=cv2.FONT_ITALIC#设置字体的样式，可以改成其他的opencv预定义字体样式
 
BASE_FONT_SIZE=23#这个可能是cv2.putText函数中，fontScale=1的时候，字体实际的像素大小

TITLE="fully-connected-first-CNN"#显示整个图像的标题
TITLE_SIZE=40*OVERALL_SCALE#标题字体的大小
TITLE_COLOR=(0,0,0)#标题的颜色
 
MAX_CHANNALS=15#最多显示多少通道（cube）
MAX_NEURONS=50#最多显示多少神经元
 
BACK_COLOR=(255,255,255)#背景颜色
 
LEFT_MARGIN=40*OVERALL_SCALE#边距们。
RIGHT_MARGIN=0*OVERALL_SCALE
TOP_MARGIN=100*OVERALL_SCALE#这里上边距包含了标题所占的区域
BOTTOM_MARGIN=10*OVERALL_SCALE
 
STRING_COLOR=(30,30,30)#设置图像中除了标题之外的其他文字的颜色
STRING_LINES=2#说明文字的最大行数，用来提前确定图像中元素的大小
STRING_SIZE=14*OVERALL_SCALE#说明文字所占像素的大小，这里可能不精确。
 
STRING_IMG_MARGIN=10*OVERALL_SCALE#文字和img元素的间隔
 
IMG_HEIGHT=400*OVERALL_SCALE#方框们所占的最大空间
IMG_WIDTH=150*OVERALL_SCALE
IMG_INTERVAL=50*OVERALL_SCALE#在img的间隔部分要画上指示器
 
CUBE_MAX_WIDTH=0.6*IMG_WIDTH#设置cube元素的最大和最小值，就是即使feature很小（大），也不会显示的太小（大），这里用cube所占img元素的比例来表示
CUBE_MAX_HEIGHT=0.6*IMG_HEIGHT
CUBE_MIN_WIDTH=0.2*IMG_WIDTH
CUBE_MIN_HEIGHT=0.2*IMG_WIDTH
 
NEURONS_RADIUS=7*OVERALL_SCALE#非卷积层中的神经元的大小
NEURONS_COLOR=(100,100,100)#neuron元素的颜色
 
TOGGLE_COLOR_1=(97,97,97)#cube元素的颜色1
TOGGLE_COLOR_2=(157,157,157)#cube元素的颜色2
 
KERNEL_SCALE=4#用来缩放卷积核，让它可以在图像中不至于太小
 
STRING_FOR_NEURONS_OFFSETX=25*OVERALL_SCALE#神经元层的描述文字有了偏移量之后就会显示的好看一些
 
def get_plot_width():
    list_len=len(CNN_list)
    return list_len*IMG_WIDTH+(list_len-1)*IMG_INTERVAL
def get_plot_height():
    return IMG_HEIGHT+STRING_LINES*STRING_SIZE+STRING_IMG_MARGIN
def get_bitmap_width():
    width=LEFT_MARGIN+RIGHT_MARGIN+get_plot_width()
 
    return width
def get_bitmap_height():
    height=TOP_MARGIN+BOTTOM_MARGIN+get_plot_height()
    return height
def draw_background():
    w=int(get_bitmap_width())
    h=int(get_bitmap_height())
 
    background= np.ones((h,w,3), np.uint8)*255
    return background
def draw_title():
    background=draw_background()
    title_lenth=len(TITLE)*TITLE_SIZE
    bottom_left=(int(get_bitmap_width()/2-title_lenth/2),
                 int(TOP_MARGIN/2+TITLE_SIZE))
 
    titled=cv2.putText(background, TITLE, bottom_left, FONT_FACE, TITLE_SIZE/BASE_FONT_SIZE, (0, 0, 0),
                       lineType=cv2.LINE_AA)
    return titled
def get_w_d_max_min():#获得feature map的最大最小的宽度和高度
    w_min=sys.maxsize
    w_max=0
    h_min=sys.maxsize
    h_max=0
    for _,_,w,h in CNN_list:
        if w>w_max:w_max=w
        if w<w_min:w_min=w
        if h>h_max:h_max=h
        if h<h_min:h_min=h
    return w_max,w_min,h_max,h_min
 
 
def draw_plot():
    titled=draw_title()
 
    last_cube_list=[]#保存所有img中最后一个cube的位置和cube的大小（位置，w,h）
    is_feature_map=[]#保存所有img的元素是否是cube，有可能是neuron元素
    #获取img的起始位置们
    locations=[[LEFT_MARGIN+i*(IMG_WIDTH+IMG_INTERVAL),TOP_MARGIN+STRING_SIZE*STRING_LINES]for i in range(len(CNN_list))]
    for i in range(len(CNN_list)):
        x,y=locations[i]
        string,c,w,h=CNN_list[i]
        #画描述文字
        cv2.putText(titled,string,(int(x),int(y+STRING_SIZE)),FONT_FACE,
                    STRING_SIZE/BASE_FONT_SIZE,STRING_COLOR,lineType=cv2.LINE_AA)
        if h>1 and w >1:#画通道数和feature map的大小
            cv2.putText(titled,"%d@%dX%d"%(c,w,h),(int(x),int(y+STRING_LINES*STRING_SIZE)),FONT_FACE,
                        STRING_SIZE/BASE_FONT_SIZE,STRING_COLOR,lineType=cv2.LINE_AA)
        else:
            cv2.putText(titled, "@%d" % c, (int(x), int(y + STRING_LINES * STRING_SIZE)), FONT_FACE,
                        STRING_SIZE / BASE_FONT_SIZE, STRING_COLOR, lineType=cv2.LINE_AA)
 
 
 
        if w==1 and h==1:#w和h均是1代表要画neuron元素了
            if c>MAX_NEURONS:
                draw_neurons = MAX_NEURONS
            else:
                draw_neurons=c
            for j in range(draw_neurons):
                neuron_interval= CUBE_MAX_HEIGHT/draw_neurons
                center=(int(x+CUBE_MAX_WIDTH/2),int(j*neuron_interval+y+STRING_IMG_MARGIN+STRING_LINES*STRING_SIZE+neuron_interval/2))
                if j%2==0:
                    cv2.circle(titled,center,   int(NEURONS_RADIUS),NEURONS_COLOR,2)
                else:
                    cv2.circle(titled, center,  int(NEURONS_RADIUS),NEURONS_COLOR,-1)
            last_cube_list.append(None)
            is_feature_map.append(False)
        else:
            #否则画cube元素
            w_max, w_min, h_max, h_min = get_w_d_max_min()#获取所有的feature中最大最小的宽度和高度
            w = CUBE_MIN_WIDTH + (w - w_min) / (w_max - w_min) * (CUBE_MAX_WIDTH - CUBE_MIN_WIDTH)
            h = CUBE_MIN_HEIGHT + (h - h_min) / (h_max - h_min) * (CUBE_MAX_WIDTH - CUBE_MIN_HEIGHT)
            if c > MAX_CHANNALS:
                draw_channels = MAX_CHANNALS
            else:
                draw_channels = c
            dw = CUBE_MAX_WIDTH / draw_channels
            dh = CUBE_MAX_HEIGHT / draw_channels
            for j in range(draw_channels):
                pts = np.array([
                    [x + dw * j,        y + dh * j + STRING_LINES * STRING_SIZE + STRING_IMG_MARGIN],
                    [x + dw * j + w,    y + dh * j + STRING_LINES * STRING_SIZE + STRING_IMG_MARGIN],
                    [x + dw * j + w,    y + dh * j + STRING_LINES * STRING_SIZE + h + STRING_IMG_MARGIN],
                    [x + dw * j,        y + dh * j + STRING_LINES * STRING_SIZE + h + STRING_IMG_MARGIN],
                ], dtype=np.int)#每个cube的四个顶点
 
                if j % 2 == 0:#切换颜色
                    cv2.fillConvexPoly(titled, pts, TOGGLE_COLOR_1)
                else:
                    cv2.fillConvexPoly(titled, pts, TOGGLE_COLOR_2)
            is_feature_map.append(True)
            last_cube_list.append([[x + dw * j,
                                    y + dh * j + STRING_LINES * STRING_SIZE + STRING_IMG_MARGIN],w,h])
    ploted = titled
    # 画卷积操作示意图
    for i in range(len(is_feature_map)-1):
        if is_feature_map[i]==True and is_feature_map[i+1]==True:
 
            source1 = (last_cube_list[i][0][0] + last_cube_list[i][1] / 4,
                       last_cube_list[i][0][1] + last_cube_list[i][2] / 4)
            source2 = (last_cube_list[i][0][0] + last_cube_list[i][1]  / 4+operation_list[i][1]*KERNEL_SCALE,
                       last_cube_list[i][0][1] + last_cube_list[i][2] / 4)
            source3 = (last_cube_list[i][0][0] + last_cube_list[i][1]  / 4+operation_list[i][1]*KERNEL_SCALE,
                       last_cube_list[i][0][1] + last_cube_list[i][2] / 4+operation_list[i][2]*KERNEL_SCALE)
            source4 = (last_cube_list[i][0][0] + last_cube_list[i][1]  / 4,
                       last_cube_list[i][0][1] + last_cube_list[i][2] / 4+operation_list[i][2]*KERNEL_SCALE)
 
            source_pts=np.array([source1,source2,source3,source4],np.int)
 
            dest1 = (last_cube_list[i+1][0][0] + last_cube_list[i+1][1] / 4,
                     last_cube_list[i+1][0][1] + last_cube_list[i+1][2] / 4)
 
            cv2.fillConvexPoly(ploted,source_pts,(125,125,125))
            for p1 in source_pts:
                cv2.line( ploted,tuple(p1),(int(dest1[0]),int(dest1[1])),(99,0,0),1,lineType=cv2.LINE_AA)
 
    operation_len = len(operation_list)
    if operation_len + 1 != len(CNN_list):
        raise Exception("操作符的数量应该是CNN网络层数-1")
    for i in range(operation_len):
        string, w, h = operation_list[i]
        x = int(LEFT_MARGIN + (i + 1) * IMG_WIDTH+IMG_INTERVAL*i-IMG_WIDTH/2.5)
        y = int(TOP_MARGIN+STRING_SIZE*STRING_LINES+STRING_IMG_MARGIN+IMG_HEIGHT/2-STRING_SIZE*2-IMG_HEIGHT/4)
 
        cv2.putText(ploted,string,(x,y),FONT_FACE,
                    STRING_SIZE/BASE_FONT_SIZE,STRING_COLOR,lineType=cv2.LINE_AA)
        if w > 0 and h > 0:
            cv2.putText(ploted,"%dX%d kernel"%(w,h),(int(x),int(y+STRING_SIZE)),FONT_FACE,
                        STRING_SIZE/BASE_FONT_SIZE,STRING_COLOR,lineType=cv2.LINE_AA)
    operationed=ploted
    return operationed
 
cv2.imshow(TITLE,draw_plot())
cv2.waitKey(0)
 
print("done")
