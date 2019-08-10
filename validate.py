'''
This file will report the accuracy accordiing to the validation data
We need to know how our clustering algorithm return the class, and need to match
our defination in validation data of pore -- 0 and non-pore -- 1.

Author: Yan Gao
email: gaoy4477@gmail.com
'''
import os 
import numpy as np 
from joblib import load
import argparse

def get_args():
	parser = argparse.ArgumentParser(description='validate model')
	parser.add_argument('--model_4D', nargs="?", type=str, 
                        help='File name of saved model for 4D data')
	parser.add_argument('--model_3D', nargs="?", type=str, 
                        help='File name of saved model for 3D data')
	args = parser.parse_args()
	print(args)
	return args

def transfer(prediction):
	transfer_result = np.zeros(len(prediction))
	for i, value in enumerate(prediction):
		if value == 1:
			transfer_result[i] = 0
		else:
			transfer_result[i] = 1
	return transfer_result

# get args
args = get_args()

# This part we load validation data
filename_4D = 'validation_data_4D_3.npy'
filename_3D = 'validation_data_3D_3.npy'

filepath_4D = os.path.join(os.getcwd(), 'validation_data', filename_4D)
filepath_3D = os.path.join(os.getcwd(), 'validation_data', filename_3D)

validation_data_4D = np.load(filepath_4D, allow_pickle=True)
validation_data_3D = np.load(filepath_3D, allow_pickle=True)

data_feature_4D = np.array(list(validation_data_4D[:, 0]))
data_feature_3D = np.array(list(validation_data_3D[:, 0]))
# they share the same label
data_label = validation_data_3D[:, 1]

# This part we load model
model_4D_path = os.path.join(os.getcwd(), 'model', args.model_4D+'.model')
model_3D_path = os.path.join(os.getcwd(), 'model', args.model_3D+'.model')

model_4D_type = load(model_4D_path)
model_3D_type = load(model_3D_path)

# Here apply model to our data
prediction_4D = model_4D_type.predict(data_feature_4D)
prediction_3D = model_3D_type.predict(data_feature_3D)
# transfer the label (only if needed)
transfer_prediction_4D = transfer(prediction_4D)
transfer_prediction_3D = transfer(prediction_3D)

accuracy_4D = np.sum(transfer_prediction_4D==data_label) / len(data_label)
accuracy_3D = np.sum(transfer_prediction_3D==data_label) / len(data_label)

print('Accuracy for 4D model:', accuracy_4D)
print('Accuracy for 3D model:', accuracy_3D)





