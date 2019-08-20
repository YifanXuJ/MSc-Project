'''
This file try to segment based on GPU

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

import tensorflow as tf 

def get_args():
	parser = argparse.ArgumentParser(description='Show single results')

	parser.add_argument('--model_4D', nargs="?", type=str, 
                        help='File name of saved model for 4D data')
	parser.add_argument('--model_3D', nargs="?", type=str, 
                        help='File name of saved model for 3D data')
	# parser.add_argument('--size', nargs="?", type=int,
	# 					help='Size of features, should be 1, 3 or 5')
	parser.add_argument('--timestamp', nargs="?", type=int,
						help='Target timestamp')
	parser.add_argument('--begin_slice', nargs="?", type=int,
						help='Begin slice')
	parser.add_argument('--end_slice', nargs="?", type=int,
						help='End slice')
	parser.add_argument('--pore_4D', nargs="?", type=int,
						help='Label for pore in 4D model')
	parser.add_argument('--pore_3D', nargs="?", type=int,
						help='Label for pore in 3D model')
	args = parser.parse_args()
	print(args)
	return args


args = get_args()

# Here we set the paramater
mask_centre = (700, 810)
radius = 550

# get the path for target slice
current_path = os.getcwd()
print(current_path)
all_timestamp = content.get_folder(current_path)
sub_path = os.path.join(current_path, all_timestamp[args.timestamp])
sub_all_tif = content.get_allslice(sub_path)

sub_path_previous = os.path.join(current_path, all_timestamp[args.timestamp-1])
sub_path_next = os.path.join(current_path, all_timestamp[args.timestamp+1])


# load the model from 'model' folder
model_4D_path = os.path.join(current_path, 'model', args.model_4D+'.model')
model_3D_path = os.path.join(current_path, 'model', args.model_3D+'.model')
model_4D_type = load(model_4D_path)
model_3D_type = load(model_3D_path)

centre_4D = model_4D_type.cluster_centers_
centre_3D = model_3D_type.cluster_centers_

num_centre_3D = centre_3D.shape[0]
num_centre_4D = centre_4D.shape[0]

# we need to prepare the data for graph
# load image for 3D segmentation
image_batch, height, width = features.get_3D_structure(sub_path, args.begin_slice, args.end_slice)
image_batch_previous, _, _ = features.get_3D_structure(sub_path_previous, args.begin_slice, args.end_slice)
image_batch_next, _, _ = features.get_3D_structure(sub_path_next, args.begin_slice, args.end_slice)

print('Creat tensorflow graph...')
# create filter based on centre
# depends on different centre size, now we assume it is 3x3x3
conv_stride = [1,1,1,1,1]
# for 3D kernel
kernel_3D_list = [tf.reshape(tf.constant(i, tf.float32), (3,3,3,1,1)) for i in centre_3D]
constant_3D_list = [np.sum(i**2) for i in centre_3D]

# treat 4D convolution as the combination of 3D convolution
kernel_4D_list_1 = [tf.reshape(tf.constant(i[:27], tf.float32), (3,3,3,1,1)) for i in centre_4D]
kernel_4D_list_2 = [tf.reshape(tf.constant(i[27:54], tf.float32), (3,3,3,1,1)) for i in centre_4D]
kernel_4D_list_3 = [tf.reshape(tf.constant(i[54:81], tf.float32), (3,3,3,1,1)) for i in centre_4D]
constant_4D_list = [np.sum(i**2) for i in centre_4D]

# create graph for tensorflow -> share the same size for input
x_3D = tf.compat.v1.placeholder(tf.float32, shape=(1, args.end_slice-args.begin_slice+1, height, width, 1))

# layer for 3D data
layer_list_3D = [tf.nn.conv3d(x_3D, filter=i, strides = conv_stride, padding='SAME') for i in kernel_3D_list]
# layer for 4D data
layer_list_4D_1 = [tf.nn.conv3d(x_3D, filter=i, strides = conv_stride, padding='SAME') for i in kernel_4D_list_1]
layer_list_4D_2 = [tf.nn.conv3d(x_3D, filter=i, strides = conv_stride, padding='SAME') for i in kernel_4D_list_2]
layer_list_4D_3 = [tf.nn.conv3d(x_3D, filter=i, strides = conv_stride, padding='SAME') for i in kernel_4D_list_3]
print('Finished!')

print('Convolution...')
# run the graph
with tf.compat.v1.Session() as sess:
	print('3D segmentation...')
	result = [sess.run(i, feed_dict={x_3D:image_batch}) for i in layer_list_3D]
	print('4D segmentation...')
	result_4D_1 = [sess.run(i, feed_dict={x_3D:image_batch}) for i in layer_list_4D_1]
	result_4D_2 = [sess.run(i, feed_dict={x_3D:image_batch_previous}) for i in layer_list_4D_2]
	result_4D_3 = [sess.run(i, feed_dict={x_3D:image_batch_next}) for i in layer_list_4D_3]

print('Calculating distance...')
# reshape and calculate the distance
result_reshape = [i.reshape(args.end_slice-args.begin_slice+1, height, width) for i in result]

result_4D_1_reshape = [i.reshape(args.end_slice-args.begin_slice+1, height, width) for i in result_4D_1]
result_4D_2_reshape = [i.reshape(args.end_slice-args.begin_slice+1, height, width) for i in result_4D_2]
result_4D_3_reshape = [i.reshape(args.end_slice-args.begin_slice+1, height, width) for i in result_4D_3]

distance_list = [constant_3D_list[i]-2*result_reshape[i] for i in range(num_centre_3D)]
distance_list_4D = [constant_4D_list[i]-2*result_4D_1_reshape[i]-2*result_4D_2_reshape[i]-2*result_4D_3_reshape[i] for i in range(num_centre_4D)]
print('Finished!')

print('Segmenting...')
# we only care the mask, so create mask here
mask = np.zeros((height, width), np.uint8)
cv2.circle(mask, mask_centre, radius, 1, thickness=-1)

compare_3D = [distance_list[args.pore_3D] < distance_list[j] for j in range(num_centre_3D) if j != args.pore_3D]
compare_4D = [distance_list_4D[args.pore_4D] < distance_list_4D[j] for j in range(num_centre_4D) if j != args.pore_4D]

segment_3D = mask
for i in compare_3D:
	segment_3D = segment_3D * i

segment_4D = mask
for i in compare_4D:
	segment_4D = segment_4D * i

# inverse color for plotting
segment_inv_3D = cv2.bitwise_not(segment_3D)
segment_inv_4D = cv2.bitwise_not(segment_4D)

print('Plotting...')
# plot the picture
for index, i in enumerate(range(args.begin_slice+1, args.end_slice)):
	fig = plt.figure()
	fig.suptitle('Segment for image \n {string}'.format(string=os.path.basename(sub_all_tif[i-1])))
	ax = plt.subplot(131)
	ax.imshow(segment_inv_3D[index+1], 'gray')
	ax.set_title('3D segmentation result')

	ax = plt.subplot(132)
	ax.imshow(segment_inv_4D[index+1], 'gray')
	ax.set_title('4D segmentation result')

	ax = plt.subplot(133)
	img = cv2.imread(sub_all_tif[i-1], -1)
	ax.imshow(img, 'gray')
	ax.set_title('Original image')

plt.show()
