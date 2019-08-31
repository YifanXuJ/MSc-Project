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
filename = 'labeled_data_0025.txt'
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

# devide pore and non-pore
data_label = validation_data_4D[:, -1]
pore_index =np.argwhere(data_label==0)
non_pore_index = np.argwhere(data_label==1)
num_pore = len(pore_index)
num_non_pore = len(non_pore_index)

validation_data_4D_pore = np.zeros((num_pore, validation_data_4D.shape[1]))
validation_data_4D_non_pore = np.zeros((num_non_pore, validation_data_4D.shape[1]))
validation_data_3D_pore = np.zeros((num_pore, validation_data_3D.shape[1]))
validation_data_3D_non_pore = np.zeros((num_non_pore, validation_data_3D.shape[1]))

for index, i in enumerate(pore_index):
    validation_data_4D_pore[index] = validation_data_4D[i]
    validation_data_3D_pore[index] = validation_data_3D[i]
    
for index, i in enumerate(non_pore_index):
    validation_data_4D_non_pore[index] = validation_data_4D[i]
    validation_data_3D_non_pore[index] = validation_data_3D[i]

pore_mean_3D = np.mean(validation_data_3D_pore, axis=0)[:-1]
non_pore_mean_3D = np.mean(validation_data_3D_non_pore, axis=0)[:-1]

pore_mean_4D = np.mean(validation_data_4D_pore, axis=0)[:-1]
non_pore_mean_4D = np.mean(validation_data_4D_non_pore, axis=0)[:-1]


img_bot_pore_3D, img_mid_pore_3D, img_top_pore_3D = transfer_image(pore_mean_3D)
img_bot_nonpore_3D, img_mid_nonpore_3D, img_top_nonpore_3D = transfer_image(non_pore_mean_3D)

img_bot_pore_4D_t0, img_mid_pore_4D_t0, img_top_pore_4D_t0 = transfer_image(pore_mean_4D[:27])
img_bot_nonpore_4D_t0, img_mid_nonpore_4D_t0, img_top_nonpore_4D_t0 = transfer_image(non_pore_mean_4D[:27])

img_bot_pore_4D_t1, img_mid_pore_4D_t1, img_top_pore_4D_t1 = transfer_image(pore_mean_4D[27:54])
img_bot_nonpore_4D_t1, img_mid_nonpore_4D_t1, img_top_nonpore_4D_t1 = transfer_image(non_pore_mean_4D[27:54])

img_bot_pore_4D_t2, img_mid_pore_4D_t2, img_top_pore_4D_t2 = transfer_image(pore_mean_4D[54:])
img_bot_nonpore_4D_t2, img_mid_nonpore_4D_t2, img_top_nonpore_4D_t2 = transfer_image(non_pore_mean_4D[54:])



fig = plt.figure()
fig.suptitle('Visualisation for pore in 3D')
ax = plt.subplot(131)
ax.imshow(img_bot_pore_3D,'gray')
ax = plt.subplot(132)
ax.imshow(img_mid_pore_3D,'gray')
ax = plt.subplot(133)
ax.imshow(img_top_pore_3D,'gray')

fig = plt.figure()
fig.suptitle('Visualisation for non-pore in 3D')
ax = plt.subplot(131)
ax.imshow(img_bot_nonpore_3D,'gray')
ax = plt.subplot(132)
ax.imshow(img_mid_nonpore_3D,'gray')
ax = plt.subplot(133)
ax.imshow(img_top_nonpore_3D,'gray')

fig = plt.figure()
fig.suptitle('Visualisation for pore in 4D')
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


fig = plt.figure()
fig.suptitle('Visualisation for non-pore in 4D')
ax = plt.subplot(331)
ax.imshow(img_bot_nonpore_4D_t0,'gray')
ax = plt.subplot(332)
ax.imshow(img_mid_nonpore_4D_t0,'gray')
ax = plt.subplot(333)
ax.imshow(img_top_nonpore_4D_t0,'gray')
ax = plt.subplot(334)
ax.imshow(img_bot_nonpore_4D_t1,'gray')
ax = plt.subplot(335)
ax.imshow(img_mid_nonpore_4D_t1,'gray')
ax = plt.subplot(336)
ax.imshow(img_top_nonpore_4D_t1,'gray')
ax = plt.subplot(337)
ax.imshow(img_bot_nonpore_4D_t2,'gray')
ax = plt.subplot(338)
ax.imshow(img_mid_nonpore_4D_t2,'gray')
ax = plt.subplot(339)
ax.imshow(img_top_nonpore_4D_t2,'gray')

plt.show()










