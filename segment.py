'''
This file will select one time stamp, and apply segmentation algorithm to all the slices

Author: Yan Gao
email: gaoy4477@gmail.com
'''

import cv2
import os
import numpy as np 
import matplotlib.pyplot as plt 
import module.content as content
import module.features as features
from joblib import load
import argparse
import time

# this function get args for segmentation
def get_args():
	parser = argparse.ArgumentParser(description='Show results')

	parser.add_argument('--model_4D', nargs="?", type=str, 
                        help='File name of saved model for 4D data')
	parser.add_argument('--model_3D', nargs="?", type=str, 
                        help='File name of saved model for 3D data')
	parser.add_argument('--size', nargs="?", type=int,
						help='Size of features, should be 1, 3 or 5')
	parser.add_argument('--timestamp', nargs="?", type=str,
						help='Target timestamp')
	parser.add_argument('--pore_4D', nargs="?", type=int,
						help='Label for pore in 4D model')
	parser.add_argument('--pore_3D', nargs="?", type=int,
						help='Label for pore in 3D model')
	args = parser.parse_args()
	print(args)
	return args

# function for saving the .png file
def save_png(raw_img_path, save_folder, img_data, height, width):
	plt.figure(figsize=(height/1000, width/1000), dpi=100)
	plt.imshow(img_data, 'gray')
	plt.axis('off')
	save_path = os.path.join(save_folder, os.path.basename(raw_img_path)+'.png')
	plt.savefig(save_path, dpi=1000)
	plt.close()


def segment(path_img, save_path_4D, save_path_3D, model_4D, model_3D,
			z_index, mask, feature_index, size, pore_4D, pore_3D, keyword):
	'''
	path_img: the absolute path for specific slice
	save_path_4D: target folder to save the 4D-based segmentation result
	save_path_3D: target folder to save the 3D-based segmentation result
	model_4D: 4D-based model to cluster
	model_3D: 3D-based model to cluster
	z_index: the index for z-axis, used for plot point cloud
	feature_index: save the index for features
	size: size of used features
	'''
	start = time.time()
	# record the time
	if size == 1:
		feature_4D, feature_3D = features.get_all_features_1(path_img, feature_index, keyword)
	elif size == 3:
		feature_4D, feature_3D = features.get_all_features_3(path_img, feature_index, keyword)
	elif size == 5:
		feature_4D, feature_3D = features.get_all_features_5(path_img, feature_index, keyword)
	else:
		raise ValueError('Please input the right size, should be 1, 3 or 5.')

	print('Segmenting...')
	prediction_4D = model_4D.predict(feature_4D)
	prediction_3D = model_3D.predict(feature_3D)
	# type is numpy array
	print('Finished!')

	coordinate = mask.nonzero()
	# here need to assign the value manually. 
	# Since the classfier will return 0 and 1 randomly
	zero_point_4D_co = np.argwhere(prediction_4D==pore_4D)
	# class "1" in 4D model means pore
	zero_point_3D_co = np.argwhere(prediction_3D==pore_3D)
	# class "1" in 3D model means pore

	height, width = mask.shape
	final_img_4D = np.ones((height,width), np.uint8)
	final_img_3D = np.ones((height,width), np.uint8)
	
	# point_4D_co = []
	# point_3D_co = []
	point_4D_co = [[int(coordinate[0][i]), int(coordinate[1][i])] for i in zero_point_4D_co]
	point_3D_co = [[int(coordinate[0][i]), int(coordinate[1][i])] for i in zero_point_3D_co]
	
	for i in zero_point_4D_co:
		final_img_4D[coordinate[0][i], coordinate[1][i]] = 0
		# point_4D_co.append([int(coordinate[0][i]), int(coordinate[1][i])])
	for j in zero_point_3D_co:
		final_img_3D[coordinate[0][j], coordinate[1][j]] = 0
		# point_3D_co.append([int(coordinate[0][j]), int(coordinate[1][j])])
	# write the image data

	print('Saving results...')
	# will return the coordinate for pore, and finally will return 
	# zero_location_4D = np.argwhere(final_img_4D==0)
	zero_location_4D = np.array(point_4D_co)
	z_4D_index = np.array([z_index] * len(zero_location_4D)).reshape((len(zero_location_4D),1))
	point_coordinate_4D = np.concatenate((zero_location_4D, z_4D_index), axis=1)
	# 3D coordiante: x: point_coordinate_4D[:,0]
	#				 y: point_coordinate_4D[:,1]
	#				 z: point_coordinate_4D[:,2]

	# zero_location_3D = np.argwhere(final_img_3D==0)
	zero_location_3D = np.array(point_3D_co)
	z_3D_index = np.array([z_index] * len(zero_location_3D)).reshape((len(zero_location_3D),1))
	point_coordinate_3D = np.concatenate((zero_location_3D, z_3D_index), axis=1)
	# 3D coordiante: x: point_coordinate_3D[:,0]
	#				 y: point_coordinate_3D[:,1]
	#				 z: point_coordinate_3D[:,2]

	# Save the picture
	# Such process will lost information, just for visualization
	# call the function defined above
	save_png(path_img, save_path_4D, final_img_4D, height, width)
	save_png(path_img, save_path_3D, final_img_3D, height, width)
	end = time.time()
	print(end-start)

	return point_coordinate_4D, point_coordinate_3D


args = get_args()

# Here we set the paramater
mask_centre = (700, 810)
radius = 550
keyword = 'SHP'

current_path = os.getcwd()
print(current_path)
all_timestamp = content.get_folder(current_path, keyword)
timestamp_index = [all_timestamp.index(i) for i in all_timestamp if args.timestamp in i]
sub_path = os.path.join(current_path, all_timestamp[timestamp_index[0]])
sub_all_tif = content.get_allslice(sub_path)

# assign the target document
document_path_4D = os.path.join(os.path.dirname(sub_all_tif[0]),'segmentation_4D')
if not os.path.exists(document_path_4D):
	os.mkdir(document_path_4D)
document_path_3D = os.path.join(os.path.dirname(sub_all_tif[0]),'segmentation_3D')
if not os.path.exists(document_path_3D):
	os.mkdir(document_path_3D)

# load the model from 'model' folder
model_4D_path = os.path.join(current_path, 'model', args.model_4D+'.model')
model_3D_path = os.path.join(current_path, 'model', args.model_3D+'.model')
model_4D_type = load(model_4D_path)
model_3D_type = load(model_3D_path)

# just pick one slice to get the mask and its corresponding features index
mask, feature_index = features.get_mask(sub_all_tif[0], mask_centre, radius, args.size)

# save point result every 100 slices
group_num = 312
begin_flag = 1

print('Will segment', len(sub_all_tif), 'slices')
for index, i in enumerate(sub_all_tif[:3]):
	if begin_flag:
		point_coordinate_4D, point_coordinate_3D = segment(i, document_path_4D, document_path_3D, model_4D_type, model_3D_type, 
													       index, mask, feature_index, args.size, args.pore_4D, args.pore_3D, keyword)
		begin_flag = 0
	else:
		add_point_4D, add_point_3D = segment(i, document_path_4D, document_path_3D, model_4D_type, model_3D_type, 
											 index, mask, feature_index, args.size, args.pore_4D, args.pore_3D, keyword)
		point_coordinate_4D = np.concatenate((point_coordinate_4D, add_point_4D), axis=0)
		point_coordinate_3D = np.concatenate((point_coordinate_3D, add_point_3D), axis=0)
	if (index+1) % group_num == 0:
		# save data for every 30 slice
		begin_flag = 1
		path_4D = os.path.join(document_path_4D, 'point_data_4D_'+str(index//group_num).rjust(len(str(len(sub_all_tif)//group_num)), '0')+'.csv')
		path_3D = os.path.join(document_path_3D, 'point_data_3D_'+str(index//group_num).rjust(len(str(len(sub_all_tif)//group_num)), '0')+'.csv')
		np.savetxt(path_4D, point_coordinate_4D, delimiter=',')
		np.savetxt(path_3D, point_coordinate_3D, delimiter=',')

path_4D = os.path.join(document_path_4D, 'point_data_4D_'+str(index//group_num).rjust(len(str(len(sub_all_tif)//group_num)), '0')+'.csv')
path_3D = os.path.join(document_path_3D, 'point_data_3D_'+str(index//group_num).rjust(len(str(len(sub_all_tif)//group_num)), '0')+'.csv')
np.savetxt(path_4D, point_coordinate_4D, delimiter=',')
np.savetxt(path_3D, point_coordinate_3D, delimiter=',')













