import os
import shutil
import glob

"""
This file help organise the segmentation results from the same slice into one folder
"""

project_path = os.getcwd()

top_folder_path = os.path.join(project_path, 'new_large_clusters_rec', 'gmm', '3d', 'cluster_3')
all_files = glob.glob(os.path.join(top_folder_path, '*.png'))


# make 401 new sub-dirs
# 400-801 for gmm
# # 391 - 811 for k-means
for i in range (400, 801):
    document_path_4D = os.path.join(top_folder_path, str(i))
    os.mkdir(document_path_4D)


for seg in all_files:
    file_name = int(os.path.basename(seg)[10:14])
    dst = os.path.join(top_folder_path, str(file_name))
    shutil.move(seg, dst)


print('Finish organisation')

