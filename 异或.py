# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/5/12 10:37
@desc: 
"""

import cv2
import numpy as np
from PIL import Image

# 加载图像并转换为灰度图像
# image = cv2.imread('"D:\\python_project\\yolov5_another\\img\\smoke_000007.jpg"')
image = np.array(Image.open('D:\\python_project\\yolov5_another\\img\\smoke_000007.jpg'))
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用边缘检测算法（例如Canny边缘检测）
edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)

# 将边缘结果与原始图像进行异或运算
enhanced_image = cv2.bitwise_xor(image, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

# 可选：对异或结果进行二值化
threshold_value = 100
_, binary_image = cv2.threshold(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY), threshold_value, 255, cv2.THRESH_BINARY)

# 显示图像
cv2.imshow('Original Image', image)
cv2.imshow('Enhanced Image', enhanced_image)
cv2.imshow('Binary Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()