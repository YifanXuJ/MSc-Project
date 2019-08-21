'''
This file will transfer the .png to .tif under its directory

Author: Yan Gao
email: gaoy4477@gmail.com
'''
from PIL import Image
import os

# give the path for .png image
path = '/Users/gavin/MSc-Project/SHP15_T113_0025/segmentation_3D'

# find all .png file
all_files = os.listdir(path)
all_files.sort()
all_png = [i for i in all_files if '.png' in i]

save_path = os.path.join(path, 'all_tif')
if not os.path.exists(save_path):
	os.mkdir(save_path)

for index, i in enumerate(all_png):
	if (index+1) % 100 == 0:
		print(index+1)
	png = Image.open(os.path.join(path, i))
	png.save(os.path.join(save_path, i[:-4]))
	png.close()