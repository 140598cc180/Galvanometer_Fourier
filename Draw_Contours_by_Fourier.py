import cv2
import numpy as np
import scipy.interpolate as interpolate
import turtle
'''
    作者：TJU_cc
    日期：2024/3/12
    用途：1.读取图片全部轮廓并存储在contours数组中
         2.依据轮廓长度选取L/10+1个离散采样点
         3.计算傅里叶描述子，即计算出离散点实部与虚部坐标
         4.turtle函数可视化(可选)
         5.存储在文件夹中(可选)
         6.ESP32-Arduino+数模转化器操纵激光振镜1以实部电压震动,激光振镜2以虚部电压震动
         7.在void_setup函数中使用多循环以实现视觉暂留
'''
#读取图像并转换为灰度图像
image = cv2.imread('jj.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#对图像进行二值化处理
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
#查找轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#创建 turtle 对象
t = turtle.Turtle()
#设置画布大小
turtle.setup(800, 600)
print(len(contours))
tmp=0
# 绘制所有轮廓
for contour in contours:
    #对轮廓进行重采样,得到等间距的点
    contour = contour.reshape(-1, 2)
    #根据轮廓长度动态调整重采样点的数量
    length = cv2.arcLength(contour, closed=True)
    num_points = int(length / 10) + 1  # 每10个像素使用一个采样点
    #确保重采样点的数量满足样条曲线插值的要求
    if num_points <= 3:
        continue
    tmp = tmp + 1
    contour_resampled = interpolate.splprep([contour[:, 0], contour[:, 1]], s=0)[0]
    contour_resampled = np.array(interpolate.splev(np.linspace(0, 1, num_points), contour_resampled)).T
    #计算轮廓的傅里叶描述子
    contour_complex = contour_resampled[:, 0] + 1j * contour_resampled[:, 1]
    descriptors = np.fft.fft(contour_complex)
    #重构轮廓
    reconstructed_contour = np.fft.ifft(descriptors)
    with open('F:\\results\\ikun\\real_part_{}.txt'.format(tmp), 'w') as f:
        f.write(',\n'.join(str(x) for x in reconstructed_contour.real))
    with open('F:\\results\\ikun\\imag_part_{}.txt'.format(tmp), 'w') as f:
        f.write(',\n'.join(str(x) for x in reconstructed_contour.imag))
    #将 turtle 移动到起始位置
    t.penup()
    t.goto(reconstructed_contour.real[0]/2, -reconstructed_contour.imag[0]/2)
    t.pendown()
    #绘制轮廓
    for i in range(1, len(reconstructed_contour)):
        t.goto(reconstructed_contour.real[i]/2, -reconstructed_contour.imag[i]/2)
#隐藏 turtle
t.hideturtle()
#保持窗口打开
turtle.done()