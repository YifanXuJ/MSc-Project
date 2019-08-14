'''
This file return the list for all annotated points, named as 'labeled_data.txt'
This txt file will locate in current content
This txt file including following information:

Path -- coordinate -- label


Author: Yan Gao
email: gaoy4477@gmail.com
'''

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import module.content as content

# setting the parameter here
# choose specific time stamp
begin_timestamp_index = 5
end_timestamp_index = 5
# set the range for randomly choosing slice
begin_slice = 600
end_slice = 800
# set the number of slices
num_slices = 100
# set the number of points for each slice
num_points = 1

# set the mask centre and radius for each slice
mask_centre = (700, 810)
radius = 550

# set the filename
filename = 'labeled_data_test.txt'


def add_mask(centre, radius, source_image):
	height, width = source_image.shape
	circle_img = np.zeros((height,width), np.uint8)
	cv2.circle(circle_img,centre,radius,1,thickness=-1)
	masked_data = cv2.bitwise_and(source_image, source_image, mask=circle_img)
	return masked_data

def random_effective_area(masked_image):
	height, width = masked_image.shape
	flag = 1
	while(flag):
		x_co = random.randint(0, height)
		y_co = random.randint(0, width)
		if masked_image[x_co, y_co] != 0:
			flag = 0
	return x_co, y_co

# transform the coordinate 
def transform(coordinate, x_coordinate, y_coordinate):
	transformed_coordinate = [(element[1]+x_coordinate-100, element[0]+y_coordinate-100) for element in coordinate]
	# transformed_coordinate = []
	# for element in coordinate:
	# 	location = (element[1]+x_coordinate-100, element[0]+y_coordinate-100)
	# 	transformed_coordinate.append(location)
	return transformed_coordinate



current_path = os.getcwd()
print("Current path:", current_path)

all_folder = content.get_folder(current_path)
sample_timestamp = all_folder[begin_timestamp_index:(end_timestamp_index+1)]

save_folder = os.path.join(current_path, 'validation_data')
if not os.path.exists(save_folder):
	os.mkdir(save_folder)
file_path = os.path.join(save_folder, filename)

# before labeling the data, delete the old file
if os.path.exists(file_path):
	os.remove(file_path)


print('Will choose timestamp from {begin} to {end}'.format(begin=all_folder[begin_timestamp_index], end=all_folder[end_timestamp_index]))
print('Will randomly choose {:d} slices from {:d} to {:d}'.format(num_slices, begin_slice, end_slice))
print('Left click to add points, and right click to remove the mose recently added points. Note that cannot redo the last point for each slice.')

for sub_timestamp in sample_timestamp:
	sub_path = os.path.join(current_path, sub_timestamp)
	sub_all_tif = content.get_allslice(sub_path)

	# First, we label all pore
	print('Please label {:d} points for pore in each picture!'.format(num_points))
	random_slice_pore = np.random.randint(begin_slice, end_slice, num_slices)
	for index, i in enumerate(random_slice_pore):
		slice_path = sub_all_tif[i]
		slice_img = cv2.imread(slice_path, -1)
		# now get the slice image
		masked_image = add_mask(mask_centre, radius, slice_img)
		x_coordinate, y_coordinate = random_effective_area(masked_image)

		plt.imshow(masked_image[x_coordinate-100:x_coordinate+100, y_coordinate-100:y_coordinate+100], 'gray')
		plt.title('Please label {:d} points for pore! ({:d}/{:d}) \n Current slice: {str}'.format(num_points, (index+1), num_slices, str=os.path.basename(slice_path)), color='red')
		# show 400x400 area to label
		coordinate = plt.ginput(n=num_points, timeout=0)

		# note that x, y from the ginput function is oppisite to our x and y, we need to transfer it
		transformed_coordinate = transform(coordinate, x_coordinate, y_coordinate)
		
		with open(file_path, 'a') as f:
			f.writelines([slice_path, ' ', str(transformed_coordinate), ' ', '0', '\n'])
			# '0' means pore

	print('Please label {:d} points for non-pore in each picture!'.format(num_points))
	# Then we label all non-pore
	random_slice_nonpore = np.random.randint(begin_slice, end_slice, num_slices)
	for index, i in enumerate(random_slice_nonpore):
		slice_path = sub_all_tif[i]
		slice_img = cv2.imread(slice_path, -1)
		masked_image = add_mask(mask_centre, radius, slice_img)
		x_coordinate, y_coordinate = random_effective_area(masked_image)

		plt.imshow(masked_image[x_coordinate-100:x_coordinate+100, y_coordinate-100:y_coordinate+100], 'gray')
		plt.title('Please label {:d} points for non-pore! ({:d}/{:d}) \n Current slice: {str}'.format(num_points, (index+1), num_slices, str=os.path.basename(slice_path)), color='red')
		# show 400x400 area to label
		coordinate = plt.ginput(n=num_points, timeout=0)

		transformed_coordinate = transform(coordinate, x_coordinate, y_coordinate)

		with open(file_path,'a') as f:
			f.writelines([slice_path, ' ', str(transformed_coordinate), ' ', '1', '\n'])
			# '1' means non-pore

print('Finished!')



