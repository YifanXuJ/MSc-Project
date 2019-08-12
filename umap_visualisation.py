import os
import numpy as np 
import umap
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import module.train as train

import warnings


warnings.filterwarnings('ignore')

data_3D = 'training_3D_5_3x3'
data_4D = 'training_4D_5_3x3'

data_4D_path = os.path.join(os.getcwd(), 'training_data', data_4D+'.npy')
data_3D_path = os.path.join(os.getcwd(), 'training_data', data_3D+'.npy')

training_data_4D = train.load_training_data(data_4D_path)
training_data_3D = train.load_training_data(data_3D_path)

print('Using subset of the data...')
subset_training_data_4D = training_data_4D[20000:40000]
subset_training_data_3D = training_data_3D[20000:40000]

print('Embedding for 4D data...')
embedding_4D_2 = umap.UMAP(n_components=2).fit_transform(subset_training_data_4D)
embedding_4D_3 = umap.UMAP(n_components=3).fit_transform(subset_training_data_4D)
print('Finished!')
print('Embedding for 3D data...')
embedding_3D_2 = umap.UMAP(n_components=2).fit_transform(subset_training_data_3D)
embedding_3D_3 = umap.UMAP(n_components=3).fit_transform(subset_training_data_3D)
print('Finished! Ploting...')

plt.figure()
ax = plt.subplot(projection='3d')
ax.scatter(embedding_4D_3[:,0], embedding_4D_3[:,1], embedding_4D_3[:,2])
ax.set_title('3D projection for 4D data',fontsize=12,color='r')

plt.figure()
ax = plt.subplot(projection='3d')
ax.scatter(embedding_3D_3[:,0], embedding_3D_3[:,1], embedding_3D_3[:,2])
ax.set_title('3D projection for 3D data',fontsize=12,color='r')


plt.figure()
ax = plt.subplot(211)
ax.scatter(embedding_4D_2[:,0],embedding_4D_2[:,1])
ax.set_title('2D projection for 4D data',fontsize=12,color='r')
ax = plt.subplot(212)
ax.scatter(embedding_3D_2[:,0],embedding_3D_2[:,1])
ax.set_title('2D projection for 3D data',fontsize=12,color='r')
plt.show()







