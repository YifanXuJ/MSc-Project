'''
This file contains all the functions which return 4D feature and 3D feature
it has 3 types: 1x1x1, 3x3x3, 5x5x5

---> get_mask() has been changed for rectangle frame

Author: Yan Gao
email: gaoy4477@gmail.com
'''

import cv2
import os
import numpy as np

import module.content as content

# This function will return the mask based on input centre and radius
# Also, it will return the feature index based on the input parameter, mask index is used for get features
"""def get_mask(path, mask_centre, radius, size):
	print('Pick one slice to get mask and features index')
	img = cv2.imread(path, -1)
	height, width = img.shape
	mask = np.zeros((height, width), np.uint8)
	cv2.circle(mask, mask_centre, radius, 1, thickness=-1)
	# Now begin to index the features
	coordinate = mask.nonzero()
	if size == 3:
		feature_index = [[i-1, i+2, j-1, j+2] for i, j in zip(coordinate[0], coordinate[1])]
	elif size == 1:
		feature_index = [[i, j] for i, j in zip(coordinate[0], coordinate[1])]
	elif size == 5:
		feature_index = [[i-2, i+3, j-2, j+3] for i, j in zip(coordinate[0], coordinate[1])]
	else:
		raise ValueError('Have not input the correct size!')

	return mask, feature_index"""
def get_mask(path, size):
	print('Pick one slice to get mask and features index')
	img = cv2.imread(path, -1)
	height, width = img.shape
	mask = np.zeros((height, width), np.uint8)
	cv2.rectangle(mask, (282, 353), (1136, 1052), 1, -1)
	# Now begin to index the features
	coordinate = mask.nonzero()
	if size == 3:
		feature_index = [[i-1, i+2, j-1, j+2] for i, j in zip(coordinate[0], coordinate[1])]
	elif size == 1:
		feature_index = [[i, j] for i, j in zip(coordinate[0], coordinate[1])]
	elif size == 5:
		feature_index = [[i-2, i+3, j-2, j+3] for i, j in zip(coordinate[0], coordinate[1])]
	else:
		raise ValueError('Have not input the correct size!')

	return mask, feature_index
	

# This function is used to get all the features for one slice
def get_all_features_3(path, feature_index, keyword):
	# it will return 3x3x3(x3) data
	print('Current slice:', path)

	time_slice = os.path.basename(os.path.dirname(path))
	# get the time stamp for this slice -> to locate its time stamp
	root_path = os.path.dirname(os.path.dirname(path))
	# get the root path
	all_timestamp = content.get_folder(root_path, keyword)

	time_slice_index = all_timestamp.index(time_slice)
	if time_slice_index == 0:
		target_t_list = [time_slice_index + 1, time_slice_index + 2]
	elif time_slice_index == (len(all_timestamp) - 1):
		target_t_list = [time_slice_index - 2, time_slice_index - 1]
	else:
		target_t_list = [time_slice_index -1, time_slice_index + 1]
	# this 'if' argument find the previous time stamp and next time stamp for current t

	# get three timestamp: previous, now, future
	current_path = os.path.dirname(path)
	previous_path = os.path.join(root_path, all_timestamp[target_t_list[0]])
	future_path = os.path.join(root_path, all_timestamp[target_t_list[1]])

	# get all the tif content
	current_all_tif = content.get_allslice(current_path)
	previous_all_tif = content.get_allslice(previous_path)
	future_all_tif = content.get_allslice(future_path)

	# this part of code for x3 space
	location_slice_index = current_all_tif.index(path)
	if location_slice_index == 0:
		target_space_list = [location_slice_index, location_slice_index+1, location_slice_index+2]
	elif location_slice_index == (len(current_all_tif)-1):
		# totally there are 1248 slices, this is a magic number
		target_space_list = [location_slice_index-2, location_slice_index-1, location_slice_index]
	else:
		target_space_list = [location_slice_index-1, location_slice_index, location_slice_index+1]
	# this 'if' argument find the 3D space for given slice

	print('Loading 9 images...')
	img_1 = cv2.imread(current_all_tif[target_space_list[0]], -1)
	img_2 = cv2.imread(current_all_tif[target_space_list[1]], -1)
	img_3 = cv2.imread(current_all_tif[target_space_list[2]], -1)
	# three images for space
	img_4 = cv2.imread(previous_all_tif[target_space_list[0]], -1)
	img_5 = cv2.imread(previous_all_tif[target_space_list[1]], -1)
	img_6 = cv2.imread(previous_all_tif[target_space_list[2]], -1)
	img_7 = cv2.imread(future_all_tif[target_space_list[0]], -1)
	img_8 = cv2.imread(future_all_tif[target_space_list[1]], -1)
	img_9 = cv2.imread(future_all_tif[target_space_list[2]], -1)
	print('Finished!')
	# nine images for space + time

	print('Getting features...')
	# feature index has been inputted
	feature_img_1 = [img_1[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_2 = [img_2[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_3 = [img_3[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_4 = [img_4[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_5 = [img_5[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_6 = [img_6[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_7 = [img_7[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_8 = [img_8[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_9 = [img_9[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]

	# transform to the numpy array
	feature_img_1 = np.array(feature_img_1)
	feature_img_2 = np.array(feature_img_2)
	feature_img_3 = np.array(feature_img_3)
	feature_img_4 = np.array(feature_img_4)
	feature_img_5 = np.array(feature_img_5)
	feature_img_6 = np.array(feature_img_6)
	feature_img_7 = np.array(feature_img_7)
	feature_img_8 = np.array(feature_img_8)
	feature_img_9 = np.array(feature_img_9)

	# get all features
	print('Finished!')
	print('Concatenating features...')
	feature_4D = np.concatenate((feature_img_4, feature_img_5, feature_img_6,
                                 feature_img_1, feature_img_2, feature_img_3,
                                 feature_img_7, feature_img_8, feature_img_9), axis=1)
	feature_3D = np.concatenate((feature_img_1, feature_img_2, feature_img_3), axis=1)
	print('Finished!')
    
	return feature_4D, feature_3D

# This function is used to get features from known point -> for evaluation
def get_assign_features_3(path, x_coordinate, y_coordinate, keyword):
	# print('Get 3x3 features')
	# name_slice = os.path.basename(path)
	# # get the name for this slice -> to locate its location
	time_slice = os.path.basename(os.path.dirname(path))
	# get the time stamp for this slice -> to licate its time stamp
	root_path = os.path.dirname(os.path.dirname(path))
	# get the root path
	all_timestamp = content.get_folder(root_path, keyword)

	time_slice_index = all_timestamp.index(time_slice)
	if time_slice_index == 0:
		target_t_list = [time_slice_index + 1, time_slice_index + 2]
	elif time_slice_index == (len(all_timestamp) - 1):
		target_t_list = [time_slice_index - 2, time_slice_index - 1]
	else:
		target_t_list = [time_slice_index -1, time_slice_index + 1]
	# this 'if' argument find the previous time stamp and next time stamp for current t

	current_path = os.path.dirname(path)
	previous_path = os.path.join(root_path, all_timestamp[target_t_list[0]])
	future_path = os.path.join(root_path, all_timestamp[target_t_list[1]])

	current_all_tif = content.get_allslice(current_path)
	previous_all_tif = content.get_allslice(previous_path)
	future_all_tif = content.get_allslice(future_path)
	# get all the tif content

	location_slice_index = current_all_tif.index(path)
	if location_slice_index == 0:
		target_space_list = [location_slice_index, location_slice_index+1, location_slice_index+2]
	elif location_slice_index == (len(current_all_tif)-1):
		# totally there are 1248 slices, this is a magic number
		target_space_list = [location_slice_index-2, location_slice_index-1, location_slice_index]
	else:
		target_space_list = [location_slice_index-1, location_slice_index, location_slice_index+1]
	# this 'if' argument find the 3D space for given slice

	img_1 = cv2.imread(current_all_tif[target_space_list[0]], -1)
	img_2 = cv2.imread(current_all_tif[target_space_list[1]], -1)
	img_3 = cv2.imread(current_all_tif[target_space_list[2]], -1)
	# three images for space
	img_4 = cv2.imread(previous_all_tif[target_space_list[0]], -1)
	img_5 = cv2.imread(previous_all_tif[target_space_list[1]], -1)
	img_6 = cv2.imread(previous_all_tif[target_space_list[2]], -1)
	img_7 = cv2.imread(future_all_tif[target_space_list[0]], -1)
	img_8 = cv2.imread(future_all_tif[target_space_list[1]], -1)
	img_9 = cv2.imread(future_all_tif[target_space_list[2]], -1)
	# nine images for space + time

	feature_img_1 = img_1[x_coordinate-1:x_coordinate+2, y_coordinate-1:y_coordinate+2].ravel()
	feature_img_2 = img_2[x_coordinate-1:x_coordinate+2, y_coordinate-1:y_coordinate+2].ravel()
	feature_img_3 = img_3[x_coordinate-1:x_coordinate+2, y_coordinate-1:y_coordinate+2].ravel()
	feature_img_4 = img_4[x_coordinate-1:x_coordinate+2, y_coordinate-1:y_coordinate+2].ravel()
	feature_img_5 = img_5[x_coordinate-1:x_coordinate+2, y_coordinate-1:y_coordinate+2].ravel()
	feature_img_6 = img_6[x_coordinate-1:x_coordinate+2, y_coordinate-1:y_coordinate+2].ravel()
	feature_img_7 = img_7[x_coordinate-1:x_coordinate+2, y_coordinate-1:y_coordinate+2].ravel()
	feature_img_8 = img_8[x_coordinate-1:x_coordinate+2, y_coordinate-1:y_coordinate+2].ravel()
	feature_img_9 = img_9[x_coordinate-1:x_coordinate+2, y_coordinate-1:y_coordinate+2].ravel()
	# get all features

	feature_3D = np.concatenate((feature_img_1, feature_img_2, feature_img_3))
	feature_4D = np.concatenate((feature_img_4, feature_img_5, feature_img_6,
								 feature_img_1, feature_img_2, feature_img_3,
								 feature_img_7, feature_img_8, feature_img_9))

	return feature_4D, feature_3D


def get_all_features_1(path, feature_index, keyword):
	# it will return 1x1x1(x3) data
	print('Current slice:', path)

	time_slice = os.path.basename(os.path.dirname(path))
	# get the time stamp for this slice -> to licate its time stamp
	root_path = os.path.dirname(os.path.dirname(path))
	# get the root path
	all_timestamp = content.get_folder(root_path, keyword)

	time_slice_index = all_timestamp.index(time_slice)
	if time_slice_index == 0:
		target_t_list = [time_slice_index + 1, time_slice_index + 2]
	elif time_slice_index == (len(all_timestamp) - 1):
		target_t_list = [time_slice_index - 2, time_slice_index - 1]
	else:
		target_t_list = [time_slice_index -1, time_slice_index + 1]
	# this 'if' argument find the previous time stamp and next time stamp for current t

	current_path = os.path.dirname(path)
	previous_path = os.path.join(root_path, all_timestamp[target_t_list[0]])
	future_path = os.path.join(root_path, all_timestamp[target_t_list[1]])

	current_all_tif = content.get_allslice(current_path)
	previous_all_tif = content.get_allslice(previous_path)
	future_all_tif = content.get_allslice(future_path)
	# get all the tif content

	# this part of code for x1 space
	location_slice_index = current_all_tif.index(path)
	target_space_list = [location_slice_index]
	# this 'if' argument find the 3D space for given slice

	print('Loading 3 images...')
	img_1 = cv2.imread(current_all_tif[target_space_list[0]], -1)
	# one images for space
	img_2 = cv2.imread(previous_all_tif[target_space_list[0]], -1)
	# past image
	img_3 = cv2.imread(future_all_tif[target_space_list[0]], -1)
	print('Finished!')
	# future image
	# three images for space + time

	print('Getting features...')
	feature_img_1 = [img_1[i[0], i[1]] for i in feature_index]
	feature_img_2 = [img_2[i[0], i[1]] for i in feature_index]
	feature_img_3 = [img_3[i[0], i[1]] for i in feature_index]
	
	feature_img_1 = np.array(feature_img_1).reshape((len(feature_img_1),1))
	feature_img_2 = np.array(feature_img_2).reshape((len(feature_img_2),1))
	feature_img_3 = np.array(feature_img_3).reshape((len(feature_img_3),1))
	# get all features
	print('Finished!')

	print('Concatenating features...')
	feature_4D = np.concatenate((feature_img_2, feature_img_1, feature_img_3), axis=1)
	feature_3D = feature_img_1
	print('Finished!')

	return feature_4D, feature_3D

def get_assign_features_1(path, x_coordinate, y_coordinate, keyword):
	# print('Get 1x1 features')
	# name_slice = os.path.basename(path)
	# # get the name for this slice -> to locate its location
	time_slice = os.path.basename(os.path.dirname(path))
	# get the time stamp for this slice -> to licate its time stamp
	root_path = os.path.dirname(os.path.dirname(path))
	# get the root path
	all_timestamp = content.get_folder(root_path, keyword)

	time_slice_index = all_timestamp.index(time_slice)
	if time_slice_index == 0:
		target_t_list = [time_slice_index + 1, time_slice_index + 2]
	elif time_slice_index == (len(all_timestamp) - 1):
		target_t_list = [time_slice_index - 2, time_slice_index - 1]
	else:
		target_t_list = [time_slice_index -1, time_slice_index + 1]
	# this 'if' argument find the previous time stamp and next time stamp for current t

	current_path = os.path.dirname(path)
	previous_path = os.path.join(root_path, all_timestamp[target_t_list[0]])
	future_path = os.path.join(root_path, all_timestamp[target_t_list[1]])

	current_all_tif = content.get_allslice(current_path)
	previous_all_tif = content.get_allslice(previous_path)
	future_all_tif = content.get_allslice(future_path)
	# get all the tif content

	# this part of code for x1 space
	location_slice_index = current_all_tif.index(path)
	target_space_list = [location_slice_index]
	# this 'if' argument find the 3D space for given slice


	img_1 = cv2.imread(current_all_tif[target_space_list[0]], -1)
	# one images for space
	img_2 = cv2.imread(previous_all_tif[target_space_list[0]], -1)
	# past image
	img_3 = cv2.imread(future_all_tif[target_space_list[0]], -1)
	# future image
	# three images for space + time

	feature_img_1 = img_1[x_coordinate, y_coordinate]
	feature_img_2 = img_2[x_coordinate, y_coordinate]
	feature_img_3 = img_3[x_coordinate, y_coordinate]
	# get all features

	feature_3D = np.array([feature_img_1], np.uint16)
	feature_4D = np.array([feature_img_2, feature_img_1, feature_img_3], np.uint16)

	return feature_4D, feature_3D


def get_all_features_5(path, feature_index, keyword):
	# it will return 5x5x5(x3) data
	print('Current slice:', path)

	time_slice = os.path.basename(os.path.dirname(path))
	# get the time stamp for this slice -> to licate its time stamp
	root_path = os.path.dirname(os.path.dirname(path))
	# get the root path
	all_timestamp = content.get_folder(root_path, keyword)

	time_slice_index = all_timestamp.index(time_slice)
	if time_slice_index == 0:
		target_t_list = [time_slice_index + 1, time_slice_index + 2]
	elif time_slice_index == (len(all_timestamp) - 1):
		target_t_list = [time_slice_index - 2, time_slice_index - 1]
	else:
		target_t_list = [time_slice_index -1, time_slice_index + 1]
	# this 'if' argument find the previous time stamp and next time stamp for current t

	current_path = os.path.dirname(path)
	previous_path = os.path.join(root_path, all_timestamp[target_t_list[0]])
	future_path = os.path.join(root_path, all_timestamp[target_t_list[1]])

	current_all_tif = content.get_allslice(current_path)
	previous_all_tif = content.get_allslice(previous_path)
	future_all_tif = content.get_allslice(future_path)
	# get all the tif content

	# this part of code for x5 space
	location_slice_index = current_all_tif.index(path)
	if location_slice_index == 0:
		target_space_list = [location_slice_index, location_slice_index+1, location_slice_index+2, location_slice_index+3, location_slice_index+4]
	elif location_slice_index == 1:
		target_space_list = [location_slice_index-1, location_slice_index, location_slice_index+1, location_slice_index+2, location_slice_index+3]
	elif location_slice_index == (len(current_all_tif)-2):
		target_space_list = [location_slice_index-3, location_slice_index-2, location_slice_index-1, location_slice_index, location_slice_index+1]
	elif location_slice_index == (len(current_all_tif)-1):
		# totally there are 1248 slices, this is a magic number
		target_space_list = [location_slice_index-4, location_slice_index-3, location_slice_index-2, location_slice_index-1, location_slice_index]
	else:
		target_space_list = [location_slice_index-2, location_slice_index-1, location_slice_index, location_slice_index+1, location_slice_index+2]
	# this 'if' argument find the 3D space for given slice

	print('Loading 15 images...')
	img_1 = cv2.imread(current_all_tif[target_space_list[0]], -1)
	img_2 = cv2.imread(current_all_tif[target_space_list[1]], -1)
	img_3 = cv2.imread(current_all_tif[target_space_list[2]], -1)
	img_4 = cv2.imread(current_all_tif[target_space_list[3]], -1)
	img_5 = cv2.imread(current_all_tif[target_space_list[4]], -1)
	# three images for space
	img_6 = cv2.imread(previous_all_tif[target_space_list[0]], -1)
	img_7 = cv2.imread(previous_all_tif[target_space_list[1]], -1)
	img_8 = cv2.imread(previous_all_tif[target_space_list[2]], -1)
	img_9 = cv2.imread(previous_all_tif[target_space_list[3]], -1)
	img_10 = cv2.imread(previous_all_tif[target_space_list[4]], -1)
	# previous timestamp
	img_11 = cv2.imread(future_all_tif[target_space_list[0]], -1)
	img_12 = cv2.imread(future_all_tif[target_space_list[1]], -1)
	img_13 = cv2.imread(future_all_tif[target_space_list[2]], -1)
	img_14 = cv2.imread(future_all_tif[target_space_list[3]], -1)
	img_15 = cv2.imread(future_all_tif[target_space_list[4]], -1)
	print('Finished!')
	# future tumestamp
	# nine images for space + time


	print('Getting features...')
	# creat 9 list to store the result
	feature_img_1  = [img_1[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_2  = [img_2[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_3  = [img_3[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_4  = [img_4[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_5  = [img_5[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_6  = [img_6[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_7  = [img_7[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_8  = [img_8[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_9  = [img_9[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_10 = [img_10[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_11 = [img_11[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_12 = [img_12[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_13 = [img_13[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_14 = [img_14[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]
	feature_img_15 = [img_15[i[0]:i[1], i[2]:i[3]].ravel() for i in feature_index]

	# transform to the numpy array
	feature_img_1 = np.array(feature_img_1)
	feature_img_2 = np.array(feature_img_2)
	feature_img_3 = np.array(feature_img_3)
	feature_img_4 = np.array(feature_img_4)
	feature_img_5 = np.array(feature_img_5)
	feature_img_6 = np.array(feature_img_6)
	feature_img_7 = np.array(feature_img_7)
	feature_img_8 = np.array(feature_img_8)
	feature_img_9 = np.array(feature_img_9)
	feature_img_10 = np.array(feature_img_10)
	feature_img_11 = np.array(feature_img_11)
	feature_img_12 = np.array(feature_img_12)
	feature_img_13 = np.array(feature_img_13)
	feature_img_14 = np.array(feature_img_14)
	feature_img_15 = np.array(feature_img_15)
	# get all features
	print('Finished!')

	print('Concatenating features...')
	feature_4D = np.concatenate((feature_img_6,  feature_img_7,  feature_img_8,  feature_img_9,  feature_img_10,
								 feature_img_1,  feature_img_2,  feature_img_3,  feature_img_4,  feature_img_5,
								 feature_img_11, feature_img_12, feature_img_13, feature_img_14, feature_img_15), axis=1)
	feature_3D = np.concatenate((feature_img_1,  feature_img_2,  feature_img_3,  feature_img_4,  feature_img_5), axis=1)
	print('Finished!')

	return feature_4D, feature_3D



def get_assign_features_5(path, x_coordinate, y_coordinate, keyword):
	# name_slice = os.path.basename(path)
	# # get the name for this slice -> to locate its location
	# print('Get 5x5 features')
	time_slice = os.path.basename(os.path.dirname(path))
	# get the time stamp for this slice -> to licate its time stamp
	root_path = os.path.dirname(os.path.dirname(path))
	# get the root path
	all_timestamp = content.get_folder(root_path, keyword)

	time_slice_index = all_timestamp.index(time_slice)
	if time_slice_index == 0:
		target_t_list = [time_slice_index + 1, time_slice_index + 2]
	elif time_slice_index == (len(all_timestamp) - 1):
		target_t_list = [time_slice_index - 2, time_slice_index - 1]
	else:
		target_t_list = [time_slice_index -1, time_slice_index + 1]
	# this 'if' argument find the previous time stamp and next time stamp for current t

	current_path = os.path.dirname(path)
	previous_path = os.path.join(root_path, all_timestamp[target_t_list[0]])
	future_path = os.path.join(root_path, all_timestamp[target_t_list[1]])

	current_all_tif = content.get_allslice(current_path)
	previous_all_tif = content.get_allslice(previous_path)
	future_all_tif = content.get_allslice(future_path)
	# get all the tif content

	# this part of code for x5 space
	location_slice_index = current_all_tif.index(path)
	if location_slice_index == 0:
		target_space_list = [location_slice_index, location_slice_index+1, location_slice_index+2, location_slice_index+3, location_slice_index+4]
	elif location_slice_index == 1:
		target_space_list = [location_slice_index-1, location_slice_index, location_slice_index+1, location_slice_index+2, location_slice_index+3]
	elif location_slice_index == (len(current_all_tif)-2):
		target_space_list = [location_slice_index-3, location_slice_index-2, location_slice_index-1, location_slice_index, location_slice_index+1]
	elif location_slice_index == (len(current_all_tif)-1):
		# totally there are 1248 slices, this is a magic number
		target_space_list = [location_slice_index-4, location_slice_index-3, location_slice_index-2, location_slice_index-1, location_slice_index]
	else:
		target_space_list = [location_slice_index-2, location_slice_index-1, location_slice_index, location_slice_index+1, location_slice_index+2]
	# this 'if' argument find the 3D space for given slice

	img_1 = cv2.imread(current_all_tif[target_space_list[0]], -1)
	img_2 = cv2.imread(current_all_tif[target_space_list[1]], -1)
	img_3 = cv2.imread(current_all_tif[target_space_list[2]], -1)
	img_4 = cv2.imread(current_all_tif[target_space_list[3]], -1)
	img_5 = cv2.imread(current_all_tif[target_space_list[4]], -1)
	# three images for space
	img_6 = cv2.imread(previous_all_tif[target_space_list[0]], -1)
	img_7 = cv2.imread(previous_all_tif[target_space_list[1]], -1)
	img_8 = cv2.imread(previous_all_tif[target_space_list[2]], -1)
	img_9 = cv2.imread(previous_all_tif[target_space_list[3]], -1)
	img_10 = cv2.imread(previous_all_tif[target_space_list[4]], -1)
	# previous timestamp
	img_11 = cv2.imread(future_all_tif[target_space_list[0]], -1)
	img_12 = cv2.imread(future_all_tif[target_space_list[1]], -1)
	img_13 = cv2.imread(future_all_tif[target_space_list[2]], -1)
	img_14 = cv2.imread(future_all_tif[target_space_list[3]], -1)
	img_15 = cv2.imread(future_all_tif[target_space_list[4]], -1)
	# future tumestamp
	# nine images for space + time

	feature_img_1 = img_1[x_coordinate-2:x_coordinate+3, y_coordinate-2:y_coordinate+3].ravel()
	feature_img_2 = img_2[x_coordinate-2:x_coordinate+3, y_coordinate-2:y_coordinate+3].ravel()
	feature_img_3 = img_3[x_coordinate-2:x_coordinate+3, y_coordinate-2:y_coordinate+3].ravel()
	feature_img_4 = img_4[x_coordinate-2:x_coordinate+3, y_coordinate-2:y_coordinate+3].ravel()
	feature_img_5 = img_5[x_coordinate-2:x_coordinate+3, y_coordinate-2:y_coordinate+3].ravel()
	feature_img_6 = img_6[x_coordinate-2:x_coordinate+3, y_coordinate-2:y_coordinate+3].ravel()
	feature_img_7 = img_7[x_coordinate-2:x_coordinate+3, y_coordinate-2:y_coordinate+3].ravel()
	feature_img_8 = img_8[x_coordinate-2:x_coordinate+3, y_coordinate-2:y_coordinate+3].ravel()
	feature_img_9 = img_9[x_coordinate-2:x_coordinate+3, y_coordinate-2:y_coordinate+3].ravel()
	feature_img_10 = img_10[x_coordinate-2:x_coordinate+3, y_coordinate-2:y_coordinate+3].ravel()
	feature_img_11 = img_11[x_coordinate-2:x_coordinate+3, y_coordinate-2:y_coordinate+3].ravel()
	feature_img_12 = img_12[x_coordinate-2:x_coordinate+3, y_coordinate-2:y_coordinate+3].ravel()
	feature_img_13 = img_13[x_coordinate-2:x_coordinate+3, y_coordinate-2:y_coordinate+3].ravel()
	feature_img_14 = img_14[x_coordinate-2:x_coordinate+3, y_coordinate-2:y_coordinate+3].ravel()
	feature_img_15 = img_15[x_coordinate-2:x_coordinate+3, y_coordinate-2:y_coordinate+3].ravel()
	# get all features

	feature_3D = np.concatenate((feature_img_1, feature_img_2, feature_img_3, feature_img_4, feature_img_5))
	feature_4D = np.concatenate((feature_img_6, feature_img_7, feature_img_8, feature_img_9, feature_img_10,
								 feature_img_1, feature_img_2, feature_img_3, feature_img_4, feature_img_5, 
								 feature_img_11, feature_img_12, feature_img_13, feature_img_14, feature_img_15))

	return feature_4D, feature_3D


def get_3D_structure(path_timestamp, begin_slice, end_slice):
	print('Current timestamp:', path_timestamp)
	# get all the tif content
	current_all_tif = content.get_allslice(path_timestamp)

	# get the image shape
	img = cv2.imread(current_all_tif[0], -1)
	height, width = img.shape

	print('Creating image batch...')
	image_batch = np.zeros((end_slice-begin_slice+1, height, width), np.float32)

	for index, i in enumerate(current_all_tif[begin_slice-1:end_slice]):
		if (index+1) % 100 == 0:
			print(index+1)
		image_batch[index] = cv2.imread(i, -1)

	# reshape the image structure to fit the tensorflow
	image_batch = np.reshape(image_batch, (1, end_slice-begin_slice+1, height, width, 1))	
	print('Finished!')
	
	return image_batch, height, width








