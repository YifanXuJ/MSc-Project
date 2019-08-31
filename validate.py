'''
This file will report the accuracy accordiing to the validation data
We need to know how our clustering algorithm return the class, and need to match our defination in validation data of pore -- 0 and non-pore -- 1.
Also, data and model should match


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
	parser.add_argument('--filename_4D', nargs="?", type=str, 
                        help='File name of saved 4D data')
	parser.add_argument('--filename_3D', nargs="?", type=str, 
                        help='File name of saved 3D data')
	parser.add_argument('--pore_4D', nargs="?", type=str,
						help='Label for pore in 4D model')
	parser.add_argument('--pore_3D', nargs="?", type=str,
						help='Label for pore in 3D model')

	args = parser.parse_args()
	print(args)
	return args

def transfer(prediction, target_label):
	# transfer the pore from string to list
	target_label = target_label.split(',')
	target_label = [int(i) for i in target_label]
	transfer_result = np.zeros(len(prediction))
	for i, label in enumerate(prediction):
		if label in target_label:
			transfer_result[i] = 0
		else:
			transfer_result[i] = 1
	return transfer_result

# sensitivity & specificity
def metrics(prediction, true_label):
	pore_index = np.argwhere(true_label==0)
	non_pore_index = np.argwhere(true_label==1)

	true_positive = [prediction[i]==true_label[i] for i in pore_index]
	true_negative = [prediction[i]==true_label[i] for i in non_pore_index]
	false_positive = [prediction[i]!=true_label[i] for i in non_pore_index]

	sensitivity = np.sum(true_positive) / len(pore_index)
	print(len(pore_index))
	specificity = np.sum(true_negative) / len(non_pore_index)
	print(len(non_pore_index))
	precision = np.sum(true_positive) / (np.sum(true_positive) + np.sum(false_positive))
	acc = (np.sum(true_positive) + np.sum(true_negative)) / (len(pore_index) + len(non_pore_index))
	return acc, sensitivity, specificity, precision


# get args
args = get_args()


# This part we load validation data
filepath_4D = os.path.join(os.getcwd(), 'validation_data', args.filename_4D+'.npy')
filepath_3D = os.path.join(os.getcwd(), 'validation_data', args.filename_3D+'.npy')

validation_data_4D = np.load(filepath_4D)
validation_data_3D = np.load(filepath_3D)

data_feature_4D = validation_data_4D[:,:-1]
data_feature_3D = validation_data_3D[:,:-1]
# they share the same label
data_label = validation_data_3D[:, -1]

# This part we load model
model_4D_path = os.path.join(os.getcwd(), 'model', args.model_4D+'.model')
model_3D_path = os.path.join(os.getcwd(), 'model', args.model_3D+'.model')

model_4D_type = load(model_4D_path)
model_3D_type = load(model_3D_path)

# Here apply model to our data
prediction_4D = model_4D_type.predict(data_feature_4D)
prediction_3D = model_3D_type.predict(data_feature_3D)
# transfer the label (only if needed)
transfer_prediction_4D = transfer(prediction_4D, args.pore_4D)
transfer_prediction_3D = transfer(prediction_3D, args.pore_3D)

acc_4D, sen_4D, spe_4D, pre_4D = metrics(transfer_prediction_4D, data_label)
acc_3D, sen_3D, spe_3D, pre_3D = metrics(transfer_prediction_3D, data_label)

print('Accuracy for 4D model: {:f} \n Sensitivity for 4D model: {:f} \n Specificity for 4D model: {:f} \n Precision for 4D model: {:f}'.format(acc_4D, sen_4D, spe_4D, pre_4D))
print('Accuracy for 3D model: {:f} \n Sensitivity for 3D model: {:f} \n Specificity for 3D model: {:f} \n Precision for 3D model: {:f}'.format(acc_3D, sen_3D, spe_3D, pre_3D))






