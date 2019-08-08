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

def get_args():
	parser = argparse.ArgumentParser(description='Show results')

	parser.add_argument('--model_4D', nargs="?", type=str, 
                        help='File name of saved model for 4D data')
	parser.add_argument('--model_3D', nargs="?", type=str, 
                        help='File name of saved model for 3D data')
	parser.add_argument('--size', nargs="?", type=int,
						help='Size of features, should be 1, 3 or 5')
	parser.add_argument('--timestamp', nargs="?", type=int,
						help='Target timestamp')
	args = parser.parse_args()
	print(args)
	return args

def segment(path_img, save_path_4D, save_path_3D, model_4D, model_3D, mask_centre, radius, size, z_index):
	'''
	path_img: the absolute path for specific slice
	save_path_4D: target folder to save the 4D-based segmentation result
	save_path_3D: target folder to save the 3D-based segmentation result
	model_4D: 4D-based model to cluster
	model_3D: 3D-based model to cluster
	mask_centre & radius: centre and radius for mask
	size: the type for feature extraction
	z_index: the index for z-axis, used for plot point cloud
	'''
	start = time.time()
	img = cv2.imread(path_img, -1)
	height, width = img.shape
	mask = np.zeros((height, width), np.uint8)
	cv2.circle(mask, mask_centre, radius, 1, thickness=-1)

	if size == 1:
		feature_4D, feature_3D = features.get_all_features_1(path_img, mask_centre, radius)
	elif size == 3:
		feature_4D, feature_3D = features.get_all_features_3(path_img, mask_centre, radius)
	elif size == 5:
		feature_4D, feature_3D = features.get_all_features_5(path_img, mask_centre, radius)
	else:
		raise ValueError('Please input the right size, should be 1, 3 or 5.')

	print('Segmenting...')
	prediction_4D = model_4D.predict(feature_4D)
	prediction_3D = model_3D.predict(feature_3D)

	final_img_4D = np.ones((height,width), np.uint8)
	final_img_3D = np.ones((height,width), np.uint8)

	coordinate = mask.nonzero()
	total_element = len(prediction_4D)
	for i in range(total_element):
		final_img_4D[coordinate[0][i], coordinate[1][i]] = np.abs(1-prediction_4D[i])
		final_img_3D[coordinate[0][i], coordinate[1][i]] = prediction_3D[i]
	print('Finished!')
	

	print('Saving results...')
	# will return the coordinate for pore, and finally will return 

	zero_location_4D = np.argwhere(final_img_4D==0)
	z_4D_index = np.array([z_index] * len(zero_location_4D)).reshape((len(zero_location_4D),1))
	point_coordinate_4D = np.concatenate((zero_location_4D, z_4D_index), axis=1)
	# 3D coordiante: x: point_coordinate_4D[:,0]
	#				 y: point_coordinate_4D[:,1]
	#				 z: point_coordinate_4D[:,2]

	zero_location_3D = np.argwhere(final_img_3D==0)
	z_3D_index = np.array([z_index] * len(zero_location_3D)).reshape((len(zero_location_3D),1))
	point_coordinate_3D = np.concatenate((zero_location_3D, z_3D_index), axis=1)
	# 3D coordiante: x: point_coordinate_3D[:,0]
	#				 y: point_coordinate_3D[:,1]
	#				 z: point_coordinate_3D[:,2]

	# Save the picture
	# it just generate the image for segmentation. Such process will lost the information
	plt.figure(figsize=(height/1000, width/1000), dpi=100)
	plt.imshow(final_img_4D, 'gray')
	plt.axis('off')
	save_path = os.path.join(save_path_4D, os.path.basename(path_img)+'.png')
	plt.savefig(save_path, dpi=1000)
	plt.close()

	plt.figure(figsize=(height/1000, width/1000), dpi=100)
	plt.imshow(final_img_3D, 'gray')
	plt.axis('off')
	save_path = os.path.join(save_path_3D, os.path.basename(path_img)+'.png')
	plt.savefig(save_path, dpi=1000)
	plt.close()
	print('Finished!')
	end = time.time()
	print('Using time:', end-start)

	return point_coordinate_4D, point_coordinate_3D


args = get_args()
# Here we set different paramater
mask_centre = (700, 810)
radius = 550

current_path = os.getcwd()
print(current_path)
all_timestamp = content.get_folder(current_path)
sub_path = os.path.join(current_path, all_timestamp[args.timestamp])
sub_all_tif = content.get_allslice(sub_path)

# assign the target document
document_path_4D = os.path.join(os.path.dirname(sub_all_tif[0]),'segmentation_4D')
if not os.path.exists(document_path_4D):
	os.mkdir(document_path_4D)
document_path_3D = os.path.join(os.path.dirname(sub_all_tif[0]),'segmentation_3D')
if not os.path.exists(document_path_3D):
	os.mkdir(document_path_3D)

model_4D_type = load(args.model_4D)
model_3D_type = load(args.model_3D)

group_num = 100
begin_flag = 1
print('Will segment', len(sub_all_tif), 'slices')
for index, i in enumerate(sub_all_tif):
	# 
	if begin_flag:
		point_coordinate_4D, point_coordinate_3D = segment(i, document_path_4D, document_path_3D, model_4D_type, model_3D_type, mask_centre, radius, args.size, index)
		begin_flag = 0
	else:
		add_point_4D, add_point_3D = segment(i, document_path_4D, document_path_3D, model_4D_type, model_3D_type, mask_centre, radius, args.size, index)
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




















