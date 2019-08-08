'''
This file return the feature for all training points

To running this file, we need to assign the parameter of the mask (a circle area)
also, we need to set the subsampling rate

we can change the number in target_timestamp and target_slice to change the range 
of stamp and slice which is used to get the training data

Author: Yan Gao
email: gaoy4477@gmail.com
'''

import os
import numpy as np 
import random

import module.content as content
import module.features as features
from module.arg_extractor import get_args

args = get_args()

# Assign the parameter of mask here
mask_centre = (690, 792)
radius = 600
# setting the subsampling rate
subsampling_rate = args.subsampling_rate


current_path = os.getcwd()
print('Current path:', current_path)

all_timestamp = content.get_folder(current_path)
target_timestamp = all_timestamp[args.begin_time: (args.end_time+1)]
# set the target timestamp which is used to get the features

all_feature_4D = []
all_feature_3D = []


for sub_timestamp in target_timestamp:
	sub_path = os.path.join(current_path, sub_timestamp)
	print('Current time stamp:', sub_path)
	sub_all_tif = content.get_allslice(sub_path)
	target_slice = sub_all_tif[args.begin_slice: (args.end_slice+1)]
	# set the target slice picture which is used to get the features
	for each_tif in target_slice:
		if args.size == 1:
			feature_4D, feature_3D = features.get_all_features_1(each_tif, mask_centre, radius)
		elif args.size == 3:
			feature_4D, feature_3D = features.get_all_features_3(each_tif, mask_centre, radius)
		elif args.size == 5:
			feature_4D, feature_3D = features.get_all_features_5(each_tif, mask_centre, radius)
		else:
			raise ValueError('Please set the right size type!')
		total_num_samples = len(feature_3D)
		sub_num_samples = int(total_num_samples * subsampling_rate)
		print('Shuffling and subsampling...')
		print('subsampling rate:', subsampling_rate)
		random.shuffle(feature_4D)
		random.shuffle(feature_3D)
		all_feature_4D.append(feature_4D[:sub_num_samples])
		all_feature_3D.append(feature_3D[:sub_num_samples])
		print('Finished!')

training_data_4D = np.array(all_feature_4D)
training_data_3D = np.array(all_feature_3D)

if os.path.exists(args.file_name_3D):
	os.remove(args.file_name_3D)
if os.path.exists(args.file_name_4D):
	os.remove(args.file_name_3D)
# before saving the data, delete the old data

np.save(args.file_name_4D, training_data_4D)
np.save(args.file_name_3D, training_data_3D)
