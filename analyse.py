import numpy as np 
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings('ignore')


raw_training_data_4D = np.load('training_data_4D_3.npy')
raw_training_data_3D = np.load('training_data_3D_3.npy')

num_sample_groups = raw_training_data_4D.shape[0]
num_samples = raw_training_data_4D.shape[1]
num_dim_4D = raw_training_data_4D.shape[2]
num_dim_3D = raw_training_data_3D.shape[2]

training_data_4D = np.zeros((num_sample_groups*num_samples, num_dim_4D))
training_data_3D = np.zeros((num_sample_groups*num_samples, num_dim_3D))

for i in range(num_sample_groups):
    training_data_4D[i*num_samples:(i+1)*num_samples,:] = raw_training_data_4D[i]
    training_data_3D[i*num_samples:(i+1)*num_samples,:] = raw_training_data_3D[i]

print('Using subset of the data...')
subset_training_data_4D = training_data_4D[0:20000]
subset_training_data_3D = training_data_3D[0:20000]

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


# X_embedded_4D = TSNE(n_components=2).fit_transform(subset_training_data_4D)
# X_embedded_3D = TSNE(n_components=2).fit_transform(subset_training_data_3D)

# plt.scatter(X_embedded_4D[:,0],X_embedded_4D[:,1])
# plt.scatter(X_embedded_3D[:,0],X_embedded_3D[:,1])
