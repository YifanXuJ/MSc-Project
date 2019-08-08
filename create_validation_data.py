'''
This file return the feature for all annotated points
Before running this file, we need to create 'labeled_data.txt' first
and just put labeled_data.txt at the same folder
This file will return two files, which correspond to two kinds of data and its label

Author: Yan Gao
email: gaoy4477@gmail.com
'''
import re
import numpy as np 
import os

import module.features as features

with open('./labeled_data.txt', 'r') as f:
	data = f.readlines()

validation_data_4D_1 = []
validation_data_3D_1 = []
validation_data_4D_3 = []
validation_data_3D_3 = []
validation_data_4D_5 = []
validation_data_3D_5 = []

for element in data:
	target_str = element.split()
	target_path = target_str[0]
	# extract the path
	target_class = int(target_str[-1])
	# extract the class
	co_string = str(target_str[1:-1])
	result = re.findall(r"\d+\.?\d*", co_string)
	total_points = len(result) // 2
	for i in range(total_points):
		print('Current picture: ', target_path)
		feature_4D_1, feature_3D_1 = features.get_assign_features_1(target_path, int(float(result[2*i])), int(float(result[2*i+1])))
		feature_4D_3, feature_3D_3 = features.get_assign_features_3(target_path, int(float(result[2*i])), int(float(result[2*i+1])))
		feature_4D_5, feature_3D_5 = features.get_assign_features_5(target_path, int(float(result[2*i])), int(float(result[2*i+1])))

		feature_4D_1_class = [feature_4D_1, target_class]
		feature_3D_1_class = [feature_3D_1, target_class]
		validation_data_4D_1.append(feature_4D_1_class)
		validation_data_4D_1.append(feature_3D_1_class)

		feature_4D_3_class = [feature_4D_3, target_class]
		feature_3D_3_class = [feature_3D_3, target_class]
		validation_data_4D_3.append(feature_4D_3_class)
		validation_data_4D_3.append(feature_3D_3_class)

		feature_4D_5_class = [feature_4D_5, target_class]
		feature_3D_5_class = [feature_3D_5, target_class]
		validation_data_4D_5.append(feature_4D_5_class)
		validation_data_4D_5.append(feature_3D_5_class)

validation_data_4D_1 = np.array(validation_data_4D_1)
validation_data_3D_1 = np.array(validation_data_3D_1)

validation_data_4D_3 = np.array(validation_data_4D_3)
validation_data_3D_3 = np.array(validation_data_3D_3)

validation_data_4D_5 = np.array(validation_data_4D_5)
validation_data_3D_5 = np.array(validation_data_3D_5)

if os.path.exists("./validation_data_4D_1.npy"):
	os.remove("./validation_data_4D_1.npy")
if os.path.exists("./validation_data_3D_1.npy"):
	os.remove("./validation_data_3D_1.npy")
# before saving the data, delete the old data

if os.path.exists("./validation_data_4D_3.npy"):
	os.remove("./validation_data_4D_3.npy")
if os.path.exists("./validation_data_3D_3.npy"):
	os.remove("./validation_data_3D_3.npy")
# before saving the data, delete the old data

if os.path.exists("./validation_data_4D_5.npy"):
	os.remove("./validation_data_4D_5.npy")
if os.path.exists("./validation_data_3D_5.npy"):
	os.remove("./validation_data_3D_5.npy")
# before saving the data, delete the old data

np.save('validation_data_4D_1.npy', validation_data_4D_1)
np.save('validation_data_3D_1.npy', validation_data_3D_1)
np.save('validation_data_4D_3.npy', validation_data_4D_3)
np.save('validation_data_3D_3.npy', validation_data_3D_3)
np.save('validation_data_4D_5.npy', validation_data_4D_5)
np.save('validation_data_3D_5.npy', validation_data_3D_5)










