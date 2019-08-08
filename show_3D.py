'''
This file will show the 3D structure of the distribution of pore for one time stamp
Using matplotlib to show the point cloud

Author: Yan Gao
email: gaoy4477@gmail.com
'''
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def show(file_path):
	# assign the target path for loading data
	# assume we have known the file path
	data = np.load(file_path)
	
	fig=plt.figure(dpi=200)
	ax=fig.add_subplot(projection='3d')
	plt.title('point cloud')
	ax.scatter(data[:,0],data[:,1],data[:,2],c='black',marker='.',s=0.1,linewidth=0,alpha=1,cmap='spectral')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	plt.show()

show('/Users/gavin/MSc-Project/SHP15_T113_0025/segmentation_4D/point_data_4D.npy')