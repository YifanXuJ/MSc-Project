'''
This file is used to find the appropriate mask centre and radius

Author: Yan Gao
email: gaoy4477@gmail.com
'''

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import module.content as content

centre = (700, 810)
radius = 550


path = '/Users/gavin/MSc-Project/SHP15_T113_0025/SHP15_T113__0025_0001.rec.16bit.tif'
img = cv2.imread(path, -1)
height, width = img.shape
circle_img = np.zeros((height, width), np.uint8)
cv2.circle(circle_img, centre, radius, 1, thickness=-1)
masked_data = cv2.bitwise_and(img, img, mask=circle_img)

plt.imshow(masked_data, 'gray')
plt.show()