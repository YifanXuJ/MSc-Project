'''
This file includes the different clustering algorithms.

Just call the function, and it will return the model in the current directory.

Author: Yan Gao
email: gaoy4477@gmail.com
'''

import cv2
import os
import matplotlib.pyplot as plt 
import numpy as np 
import time
import sklearn

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN

from joblib import dump, load

def load_training_data(path):
	raw_training_data = np.load(path)
	num_sample_groups = raw_training_data.shape[0]
	num_samples = raw_training_data.shape[1]
	num_dim = raw_training_data.shape[2]

	training_data = np.zeros((num_sample_groups*num_samples, num_dim))
	for i in range(num_sample_groups):
		training_data[i*num_samples:(i+1)*num_samples, :] = raw_training_data[i] 
	return training_data

def kmeans_algorithm(num_cluster, training_data, filename):
	# filename should be a string
	if not isinstance(filename, str):
		raise ValueError('filename should be a string!')

	print('Running...')
	start = time.time()
	kmeans = KMeans(n_clusters=num_cluster).fit(training_data)
	end = time.time()
	print('Run time:', end-start)

	print('Saving model, please wait...')
	if os.path.exists(filename):
		os.remove(filename)
	dump(kmeans, filename)

	print('Finished!')

def mini_batch_kmeans_algorithm(num_cluster, training_data, filename):
	# filename should be a string
	if not isinstance(filename, str):
		raise ValueError('filename should be a string!')

	print('Running...')
	start = time.time()
	minibatch_kmeans = MiniBatchKMeans(n_clusters=num_cluster, batch_size=100000).fit(training_data)
	end = time.time()
	print('Run time:', end-start)

	print('Saving model, please wait...')
	if os.path.exists(filename):
		os.remove(filename)
	dump(minibatch_kmeans, filename)

	print('Finished!')

def gmm(num_components, covariance_type, training_data, filename):
	if not isinstance(filename, str):
		raise ValueError('filename should be a string!')

	print('Running...')
	start = time.time()
	gmm = GaussianMixture(n_components=num_components, covariance_type=covariance_type).fit(training_data)
	end = time.time()
	print('Run time:', end-start)

	print('Saving model, please wait...')
	if os.path.exists(filename):
		os.remove(filename)
	dump(gmm, filename)

	print('Finished!')

def mean_shift(training_data, filename, input_bandwidth):
	if not isinstance(filename, str):
		raise ValueError('filename should be a string!')
	print('Running...')
	start = time.time()
	meanshift = MeanShift(bandwidth=input_bandwidth).fit(training_data)
	end = time.time()
	print('Run time:', end-start)
	print(meanshift.cluster_centers_.shape)

	print('Saving model, please wait...')
	if os.path.exists(filename):
		os.remove(filename)
	dump(meanshift, filename)

	print('Finished!')


# def dbscan(training_data, filename):
# 	if not ßisinstance(filename, str):
# 		raise ValueError('filename should be a string!')
# 	start = time.time()
# 	dbscan = DBSCAN(eps=3, min_samples=2).fit(training_data)
# 	end = time.time()
# 	print('Run time:', end-start)

# 	print('Saving model, please wait...')
# 	if os.path.exists(filename):
# 		os.remove(filename)
# 	dump(dbscan, filename)

# 	print('Finished!')



















