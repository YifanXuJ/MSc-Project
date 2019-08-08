'''
This file will do the training process.
just modify the parameter in this file to change the model.
It will return the model in current directory.


Author: Yan Gao
email: gaoy4477@gmail.com
'''

import numpy as np 
import module.train as train
import argparse

def get_args():
	parser = argparse.ArgumentParser(description='Training process')
	parser.add_argument('--training_data_3D', nargs="?", type=str, 
    					help='File name of saved 3D feature')
	parser.add_argument('--training_data_4D', nargs="?", type=str, 
                        help='File name of saved 4D feature')
	parser.add_argument('--model_type', nargs="?", type=str, 
                        help='Type of model, shoulde be kmeans, mini_batch_kmeans or gmm')
	parser.add_argument('--num_cluster', nargs="?", type=int, default=2,
    					help='Number of clusters')
	parser.add_argument('--covariance_type', nargs="?", type=str, default='full',
    					help='Type of covariance for gmm model')
	parser.add_argument('--model_4D', nargs="?", type=str, 
                        help='File name of saved model for 4D data')
	parser.add_argument('--model_3D', nargs="?", type=str, 
                        help='File name of saved model for 3D data')
	args = parser.parse_args()
	print(args)
	return args

args = get_args()

raw_training_data_4D = np.load(args.training_data_4D)
raw_training_data_3D = np.load(args.training_data_3D)

num_sample_groups = raw_training_data_4D.shape[0]
num_samples = raw_training_data_4D.shape[1]
num_dim_4D = raw_training_data_4D.shape[2]
num_dim_3D = raw_training_data_3D.shape[2]

training_data_4D = np.zeros((num_sample_groups*num_samples, num_dim_4D))
training_data_3D = np.zeros((num_sample_groups*num_samples, num_dim_3D))

for i in range(num_sample_groups):
    training_data_4D[i*num_samples:(i+1)*num_samples,:] = raw_training_data_4D[i]
    training_data_3D[i*num_samples:(i+1)*num_samples,:] = raw_training_data_3D[i]

print('Number of samples:', num_sample_groups*num_samples)

if args.model_type == 'kmeans':
	# kmeans 
	# train.kmeans_algorithm(num_cluster, training_data, filename)
	print('kmeans for 4D data')
	train.kmeans_algorithm(args.num_cluster, training_data_4D, args.model_4D)
	print('kmeans for 3D data')
	train.kmeans_algorithm(args.num_cluster, training_data_3D, args.model_3D)
elif args.model_type == 'mini_batch_kmeans':
	# mini-batch kmeans
	# train.mini_batch_kmeans_algorithm(num_cluster, training_data, filename)
	print('mini-batch kmeans for 4D data')
	train.mini_batch_kmeans_algorithm(args.num_cluster, training_data_4D, args.model_4D)
	print('mini-batch kmeans for 3D data')
	train.mini_batch_kmeans_algorithm(args.num_cluster, training_data_3D, args.model_3D)
elif args.model_type == 'gmm':
	# GMM
	# train.gmm(num_components, covariance_type, training_data, filename)
	# There are four covariance_type: {‘full’, ‘tied’, ‘diag’, ‘spherical’}
	# all the parameters need to be assigned
	print('GMM for 4D data')
	train.gmm(args.num_cluster, args.covariance_type, training_data_4D, args.model_4D)
	print('GMM for 3D data')
	train.gmm(args.num_cluster, args.covariance_type, training_data_3D, args.model_3D)
else:
	raise ValueError('Please input the correct type name!')








