# Create training data
python create_training_data.py --begin_time 5 --end_time 5 --begin_slice 600 --end_slice 799 --filename_3D training_3D_5_3x3 --filename_4D training_4D_5_3x3 --size 3


# Create validation data
python create_validation_data.py 


# Training 3x3x3 dataset
# 2 clusters
python training.py --data_3D training_3D_5_3x3 --data_4D training_4D_5_3x3 --model_type mini_batch_kmeans --num_cluster 2 --name_4D mini_kmeans_4D_2_3x3 --name_3D mini_kmeans_3D_2_3x3


# Validate data
python validate.py --model_4D mini_kmeans_4D_2_3x3 --model_3D mini_kmeans_3D_2_3x3


# Segment slices
python segment.py --model_4D mini_kmeans_4D_2_3x3 --model_3D mini_kmeans_3D_2_3x3 --size 3 --timestamp 5


# visualisation centre
python centre_visualisation.py --model_4D mini_kmeans_4D_2_3x3 --model_3D mini_kmeans_3D_2_3x3 --type mini_batch_kmeans