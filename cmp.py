import os
import cv2
import module.content as content
import numpy as np 

path_1 = '/Users/gavin/MSc-Project/SHP15_T113_0025/segmentation_4D'
path_2 = '/Users/gavin/MSc-Project/results/segmentation_4D'

all_png_1 = content.get_allsegment(path_1)
all_png_2 = content.get_allsegment(path_2)

img1 = cv2.imread(all_png_1[1], 0)
img2 = cv2.imread(all_png_2[0], 0)

compare = img1 == img2
index = np.argwhere(compare==0)