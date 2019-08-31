'''
This file will visualise the average grayscale value for pore and non-pore point
load data from validation data


Author: Yan Gao
email: gaoy4477@gmail.com
'''
import re
import numpy as np 
import os
import module.features as features
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('MacOSX')

# this function will transfer 9x1 array to 3x3 array with uint16 type
def get_image(array):
	dim = int(np.sqrt(len(array)))
	convert = array.astype(np.uint16)
	img = convert.reshape((dim, dim))
	return img

# transfer a 27-dim array to 3 images
def transfer_image(array):
	length = int(len(array))
	div_length = length // 3
	img_bot = get_image(array[:div_length])
	img_mid = get_image(array[div_length:2*div_length])
	img_top = get_image(array[2*div_length:])
	return img_bot, img_mid, img_top


# here, need to give the full name of target text file (with the '.txt')
# change it for different labeled data
filename = 'artifact_0025.txt'
keyword = 'SHP'
feature_size = 3

print('Loading text file...')
filepath = os.path.join(os.getcwd(), 'validation_data', filename)

with open(filepath, 'r') as f:
	data = f.readlines()

validation_data_4D = []
validation_data_3D = []

for element in data:
	target_str = element.split()
	target_path = os.path.join(os.getcwd(), os.path.basename(os.path.dirname(target_str[0])), os.path.basename(target_str[0]))
	# extract the path
	target_class = int(target_str[-1])
	# extract the class
	co_string = str(target_str[1:-1])
	result = re.findall(r"\d+\.?\d*", co_string)
	total_points = len(result) // 2
	for i in range(total_points):
		print('Current picture: ', target_path)
		# get features
		if feature_size == 3:
			feature_4D, feature_3D = features.get_assign_features_3(target_path, int(float(result[2*i])), int(float(result[2*i+1])), keyword)
		elif feature_size == 1:
			feature_4D, feature_3D = features.get_assign_features_1(target_path, int(float(result[2*i])), int(float(result[2*i+1])), keyword)
		elif feature_size == 5:
			feature_4D, feature_3D = features.get_assign_features_5(target_path, int(float(result[2*i])), int(float(result[2*i+1])), keyword)
		else:
			raise ValueError('Please give the correct size!')

		feature_4D_class = np.append(feature_4D, target_class)
		feature_3D_class = np.append(feature_3D, target_class)
		
		validation_data_4D.append(feature_4D_class)
		validation_data_3D.append(feature_3D_class)

# transfer its format
validation_data_4D = np.array(validation_data_4D)
validation_data_3D = np.array(validation_data_3D)


# all are artifact
artifact_4D = validation_data_4D[:, :-1]
artifact_3D = validation_data_3D[:, :-1]

artifact_mean_3D = np.mean(artifact_3D, axis=0)
artifact_mean_4D = np.mean(artifact_4D, axis=0)



img_bot_pore_3D, img_mid_pore_3D, img_top_pore_3D = transfer_image(artifact_mean_3D)

img_bot_pore_4D_t0, img_mid_pore_4D_t0, img_top_pore_4D_t0 = transfer_image(artifact_mean_4D[:27])

img_bot_pore_4D_t1, img_mid_pore_4D_t1, img_top_pore_4D_t1 = transfer_image(artifact_mean_4D[27:54])

img_bot_pore_4D_t2, img_mid_pore_4D_t2, img_top_pore_4D_t2 = transfer_image(artifact_mean_4D[54:])



fig = plt.figure()
fig.suptitle('Visualisation for artifact in 3D')
ax = plt.subplot(131)
ax.imshow(img_bot_pore_3D,'gray')
ax = plt.subplot(132)
ax.imshow(img_mid_pore_3D,'gray')
ax = plt.subplot(133)
ax.imshow(img_top_pore_3D,'gray')


fig = plt.figure()
fig.suptitle('Visualisation for artifact in 4D')
ax = plt.subplot(331)
ax.imshow(img_bot_pore_4D_t0,'gray')
ax = plt.subplot(332)
ax.imshow(img_mid_pore_4D_t0,'gray')
ax = plt.subplot(333)
ax.imshow(img_top_pore_4D_t0,'gray')
ax = plt.subplot(334)
ax.imshow(img_bot_pore_4D_t1,'gray')
ax = plt.subplot(335)
ax.imshow(img_mid_pore_4D_t1,'gray')
ax = plt.subplot(336)
ax.imshow(img_top_pore_4D_t1,'gray')
ax = plt.subplot(337)
ax.imshow(img_bot_pore_4D_t2,'gray')
ax = plt.subplot(338)
ax.imshow(img_mid_pore_4D_t2,'gray')
ax = plt.subplot(339)
ax.imshow(img_top_pore_4D_t2,'gray')


plt.show()










