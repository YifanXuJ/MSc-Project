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

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift

from joblib import dump, load

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



















