'''
This file will return the visualisation based on UMAP
If we assign the centre from kmeans, it can show the location of the centre in the transformed space

Author: Yan Gao
email: gaoy4477@gmail.com
'''
import os
import numpy as np 
import umap
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import module.train as train
from joblib import load

import warnings

# will ignore some warnings
warnings.filterwarnings('ignore')

# if dont want to show the centre, just assign flag_show_centre = False
# show centre only for kmeans or mini_batch_kmeans algorithm
flag_show_centre = True
model_4D = 'mini_kmeans_4D_2_3x3'
model_3D = 'mini_kmeans_3D_2_3x3'

# Load the data
data_3D = 'training_3D_5_3x3'
data_4D = 'training_4D_5_3x3'

data_4D_path = os.path.join(os.getcwd(), 'training_data', data_4D+'.npy')
data_3D_path = os.path.join(os.getcwd(), 'training_data', data_3D+'.npy')

print('Loading data...')
training_data_4D = train.load_training_data(data_4D_path)
training_data_3D = train.load_training_data(data_3D_path)

print('Using subset of the data...')
subset_training_data_4D = training_data_4D[20000:40000]
subset_training_data_3D = training_data_3D[20000:40000]

print('Embedding for 4D data...')
embedding_4D_2_model = umap.UMAP(n_components=2).fit(subset_training_data_4D)
embedding_4D_3_model = umap.UMAP(n_components=3).fit(subset_training_data_4D)
print('Finished!')
print('Embedding for 3D data...')
embedding_3D_2_model = umap.UMAP(n_components=2).fit(subset_training_data_3D)
embedding_3D_3_model = umap.UMAP(n_components=3).fit(subset_training_data_3D)
print('Finished!')

# if want to show the centre, will run following code
if flag_show_centre:
	model_4D_path = os.path.join(os.getcwd(), 'model', model_4D+'.model')
	model_3D_path = os.path.join(os.getcwd(), 'model', model_3D+'.model')

	model_4D_type = load(model_4D_path)
	model_3D_type = load(model_3D_path)

	centre_4D = model_4D_type.cluster_centers_
	centre_3D = model_3D_type.cluster_centers_
	
	num_centre_4D = centre_4D.shape[0]
	num_centre_3D = centre_3D.shape[0]

	# Assume we have known just 2 centres, we can change this assumption
	# project 4D centre
	centre_4D_list_3D = []
	centre_4D_list_2D = []

	centre_3D_list_3D = []
	centre_3D_list_2D = []
	for i in range(num_centre_4D):
		centre_4D_list_3D.append(embedding_4D_3_model.transform(centre_4D[i].reshape(1,-1)))
		centre_4D_list_2D.append(embedding_4D_2_model.transform(centre_4D[i].reshape(1,-1)))

	for i in range(num_centre_3D):
		centre_3D_list_3D.append(embedding_3D_3_model.transform(centre_3D[i].reshape(1,-1)))
		centre_3D_list_2D.append(embedding_3D_2_model.transform(centre_3D[i].reshape(1,-1)))


# transform the data
embedding_4D_2 = embedding_4D_2_model.transform(subset_training_data_4D)
embedding_4D_3 = embedding_4D_3_model.transform(subset_training_data_4D)
embedding_3D_2 = embedding_3D_2_model.transform(subset_training_data_3D)
embedding_3D_3 = embedding_3D_3_model.transform(subset_training_data_3D)


print('Plotting...')
plt.figure()
ax = plt.subplot(projection='3d')
ax.scatter(embedding_4D_3[:,0], embedding_4D_3[:,1], embedding_4D_3[:,2], alpha=0.01)
if flag_show_centre:
	for i in range(num_centre_4D):
		ax.scatter(centre_4D_list_3D[i][:,0], centre_4D_list_3D[i][:,1], centre_4D_list_3D[i][:,2], color='red')
ax.set_title('3D projection for 4D data',fontsize=12,color='r')

plt.figure()
ax = plt.subplot(projection='3d')
ax.scatter(embedding_3D_3[:,0], embedding_3D_3[:,1], embedding_3D_3[:,2], alpha=0.01)
if flag_show_centre:
	for i in range(num_centre_3D):
		ax.scatter(centre_3D_list_3D[i][:,0], centre_3D_list_3D[i][:,1], centre_3D_list_3D[i][:,2], color='red')
ax.set_title('3D projection for 3D data',fontsize=12,color='r')


plt.figure()
ax = plt.subplot(211)
ax.scatter(embedding_4D_2[:,0],embedding_4D_2[:,1])
if flag_show_centre:
	for i in range(num_centre_4D):
		ax.scatter(centre_4D_list_2D[i][:,0], centre_4D_list_2D[i][:,1], color='red')
ax.set_title('2D projection for 4D data',fontsize=12,color='r')
ax = plt.subplot(212)
ax.scatter(embedding_3D_2[:,0],embedding_3D_2[:,1])
if flag_show_centre:
	for i in range(num_centre_3D):
		ax.scatter(centre_3D_list_2D[i][:,0], centre_3D_list_2D[i][:,1], color='red')
ax.set_title('2D projection for 3D data',fontsize=12,color='r')


plt.show()







