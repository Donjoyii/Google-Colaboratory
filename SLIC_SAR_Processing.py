import cv2
import numpy as np
from google.colab.patches import cv2_imshow

img = cv2.imread("river_2.bmp")
### SLIC 算法
# 初始化slic项，超像素平均尺寸20（默认为10），平滑因子20
slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=20, ruler=20.0) 
slic.iterate(100)     # 迭代次数，越大效果越好
mask_slic = slic.getLabelContourMask()     # 获取Mask，超像素边缘Mask==1
label_slic = slic.getLabels()     # 获取超像素标签
number_slic = slic.getNumberOfSuperpixels()     # 获取超像素数目
mask_inv_slic = cv2.bitwise_not(mask_slic)  
img_slic = cv2.bitwise_and(img, img, mask=mask_inv_slic) #在原图上绘制超像素边界

color_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
color_img[:] = (0, 255 , 0)
result_ = cv2.bitwise_and(color_img, color_img, mask=mask_slic)
result = cv2.add(img_slic, result_)
cv2_imshow(result)
