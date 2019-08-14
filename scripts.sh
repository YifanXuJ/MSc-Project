# Create training data
python create_training_data.py --begin_time 5 --end_time 5 --begin_slice 600 --end_slice 799 --filename_3D training_3D_5_3x3 --filename_4D training_4D_5_3x3 --size 3
python create_training_data.py --begin_time 6 --end_time 6 --begin_slice 600 --end_slice 799 --filename_3D training_3D_6_3x3 --filename_4D training_4D_6_3x3 --size 3

# Annotate data
python annotation.py


# Create validation data
python create_validation_data.py 


# Training 3x3x3 dataset
# 2 clusters
python train.py --data_3D training_3D_5_3x3 --data_4D training_4D_5_3x3 --model_type mini_batch_kmeans --num_cluster 2 --name_4D mini_kmeans_4D_2_3x3 --name_3D mini_kmeans_3D_2_3x3
# 3 clusters
python train.py --data_3D training_3D_5_3x3 --data_4D training_4D_5_3x3 --model_type mini_batch_kmeans --num_cluster 3 --name_4D mini_kmeans_4D_3_3x3 --name_3D mini_kmeans_3D_3_3x3
python train.py --data_3D training_3D_6_3x3 --data_4D training_4D_6_3x3 --model_type mini_batch_kmeans --num_cluster 3 --name_4D mini_kmeans_4D_3_3x3_6 --name_3D mini_kmeans_3D_3_3x3_6

# show single slice
python show_single.py --model_4D mini_kmeans_4D_2_3x3 --model_3D mini_kmeans_3D_2_3x3 --size 3 --timestamp 5 --slice 500 --pore_4D 1 --pore_3D 1
python show_single.py --model_4D mini_kmeans_4D_3_3x3 --model_3D mini_kmeans_3D_3_3x3 --size 3 --timestamp 5 --slice 500 --pore_4D 1 --pore_3D 2

python show_single.py --model_4D mini_kmeans_4D_3_3x3_6 --model_3D mini_kmeans_3D_3_3x3_6 --size 3 --timestamp 6 --slice 600 --pore_4D 1 --pore_3D 2

# Validate data
python validate.py --model_4D mini_kmeans_4D_2_3x3 --model_3D mini_kmeans_3D_2_3x3 --filename_4D validation_data_4D_3 --filename_3D validation_data_3D_3 --pore_4D 1 --pore_3D 1
python validate.py --model_4D mini_kmeans_4D_3_3x3 --model_3D mini_kmeans_3D_3_3x3 --filename_4D validation_data_4D_3 --filename_3D validation_data_3D_3 --pore_4D 1 --pore_3D 2


# Segment slices
python segment.py --model_4D mini_kmeans_4D_2_3x3 --model_3D mini_kmeans_3D_2_3x3 --size 3 --timestamp 5 --pore_4D 1 --pore_3D 1
python segment.py --model_4D mini_kmeans_4D_3_3x3 --model_3D mini_kmeans_3D_3_3x3 --size 3 --timestamp 5 --pore_4D 1 --pore_3D 2

python segment.py --model_4D mini_kmeans_4D_3_3x3_6 --model_3D mini_kmeans_3D_3_3x3_6 --size 3 --timestamp 6 --pore_4D 1 --pore_3D 2



# visualisation centre
python visualisation_cluster_centre.py --model_4D mini_kmeans_4D_2_3x3 --model_3D mini_kmeans_3D_2_3x3 --type mini_batch_kmeans
python visualisation_cluster_centre.py --model_4D mini_kmeans_4D_3_3x3 --model_3D mini_kmeans_3D_3_3x3 --type mini_batch_kmeans


# visualisation umap
python visualisation_umap.py