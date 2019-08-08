'''
This file includes the function which shows the segmentation result
It will return 3 pictures: segmentation by 4D data, by 3D data and original data

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
	parser.add_argument('--slice', nargs="?", type=int,
						help='Target slice')
	args = parser.parse_args()
	print(args)
	return args

def show(path, model_4D, model_3D, mask_centre, radius, size):
	# path -> target slice
	# model -> kmeans, mini_batch kmeans or gmm
	# Note that here, the feature_4D and feature_3D should match the model
	img = cv2.imread(path, -1)
	height, width = img.shape
	mask = np.zeros((height, width), np.uint8)
	cv2.circle(mask, mask_centre, radius, 1, thickness=-1)

	if size == 1:
		feature_4D, feature_3D = features.get_all_features_1(path, mask_centre, radius)
	elif size == 3:
		feature_4D, feature_3D = features.get_all_features_3(path, mask_centre, radius)
	elif size == 5:
		feature_4D, feature_3D = features.get_all_features_5(path, mask_centre, radius)
	else:
		raise ValueError('Please input the right size, should be 1, 3 or 5.')

	prediction_4D = model_4D.predict(feature_4D)
	prediction_3D = model_3D.predict(feature_3D)

	final_img_4D = np.ones((height,width), np.uint8)
	final_img_3D = np.ones((height,width), np.uint8)

	print('Plotting picture...')
	coordinate = mask.nonzero()
	total_element = len(prediction_4D)
	for i in range(total_element):
		final_img_4D[coordinate[0][i], coordinate[1][i]] = np.abs(1-prediction_4D[i])
		final_img_3D[coordinate[0][i], coordinate[1][i]] = prediction_3D[i]


	plt.figure(figsize=(180, 60))
	plt.subplot(131)
	plt.imshow(final_img_4D, 'gray')
	plt.subplot(132)
	plt.imshow(final_img_3D, 'gray')
	plt.subplot(133)
	plt.imshow(img, 'gray')
	plt.show()


for i in range(total_element):
	final_img_4D[coordinate[0][i], coordinate[1][i]] = np.abs(1-prediction_4D[i])
	final_img_3D[coordinate[0][i], coordinate[1][i]] = prediction_3D[i]

plt.figure()
plt.imshow(final_img_4D, 'gray')
plt.figure()
plt.imshow(final_img_3D, 'gray')
plt.figure()
plt.imshow(img, 'gray')
plt.show()


args = get_args()
# Here we set different paramater
mask_centre = (690, 792)
radius = 600


current_path = os.getcwd()
print(current_path)
all_timestamp = content.get_folder(current_path)
sub_path = os.path.join(current_path, all_timestamp[args.timestamp])
sub_all_tif = content.get_allslice(sub_path)
path_img = sub_all_tif[args.slice]

kmeans_model_4D = load(args.model_4D)
kmeans_model_3D = load(args.model_3D)

show(path_img, kmeans_model_4D, kmeans_model_3D, mask_centre, radius, args.size)








