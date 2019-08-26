# Create training data
python create_training_data.py --begin_time 0021 --end_time 0049 --begin_slice 600 --end_slice 800 --filename_3D training_data_3D_3x3 --filename_4D training_data_4D_3x3 --size 3
python create_training_data.py --begin_time 0021 --end_time 0049 --begin_slice 600 --end_slice 800 --filename_3D training_data_3D_5x5 --filename_4D training_data_4D_5x5 --size 5
python create_training_data.py --begin_time 0021 --end_time 0049 --begin_slice 600 --end_slice 800 --filename_3D training_data_3D_1x1 --filename_4D training_data_4D_1x1 --size 1

python create_training_data.py --begin_time 0025 --end_time 0025 --begin_slice 600 --end_slice 800 --filename_3D training_data_3D_3x3_0025 --filename_4D training_data_4D_3x3_0025 --size 3
python create_training_data.py --begin_time 0025 --end_time 0025 --begin_slice 600 --end_slice 800 --filename_3D training_data_3D_5x5_0025 --filename_4D training_data_4D_5x5_0025 --size 5
python create_training_data.py --begin_time 0025 --end_time 0025 --begin_slice 600 --end_slice 800 --filename_3D training_data_3D_1x1_0025 --filename_4D training_data_4D_1x1_0025 --size 1


# Annotate data
python annotation.py
# Create validation data
python create_validation_data.py 


# Training 1x1x1 dataset
# kmeans
# 2 clusters
python train.py --data_3D training_data_3D_1x1 --data_4D training_data_4D_1x1 --model_type kmeans --num_cluster 2 --name_4D kmeans_4D_2_1x1 --name_3D kmeans_3D_2_1x1
python train.py --data_3D training_data_3D_1x1_0025 --data_4D training_data_4D_1x1_0025 --model_type kmeans --num_cluster 2 --name_4D kmeans_4D_2_1x1_0025 --name_3D kmeans_3D_2_1x1_0025
# 3 clusters
python train.py --data_3D training_data_3D_1x1 --data_4D training_data_4D_1x1 --model_type kmeans --num_cluster 3 --name_4D kmeans_4D_3_1x1 --name_3D kmeans_3D_3_1x1
python train.py --data_3D training_data_3D_1x1_0025 --data_4D training_data_4D_1x1_0025 --model_type kmeans --num_cluster 3 --name_4D kmeans_4D_3_1x1_0025 --name_3D kmeans_3D_3_1x1_0025
# 4 clusters
python train.py --data_3D training_data_3D_1x1 --data_4D training_data_4D_1x1 --model_type kmeans --num_cluster 4 --name_4D kmeans_4D_4_1x1 --name_3D kmeans_3D_4_1x1
python train.py --data_3D training_data_3D_1x1_0025 --data_4D training_data_4D_1x1_0025 --model_type kmeans --num_cluster 4 --name_4D kmeans_4D_4_1x1_0025 --name_3D kmeans_3D_4_1x1_0025
# 5 clusters
python train.py --data_3D training_data_3D_1x1 --data_4D training_data_4D_1x1 --model_type kmeans --num_cluster 5 --name_4D kmeans_4D_5_1x1 --name_3D kmeans_3D_5_1x1
python train.py --data_3D training_data_3D_1x1_0025 --data_4D training_data_4D_1x1_0025 --model_type kmeans --num_cluster 5 --name_4D kmeans_4D_5_1x1_0025 --name_3D kmeans_3D_5_1x1_0025
# 6 clusters
python train.py --data_3D training_data_3D_1x1 --data_4D training_data_4D_1x1 --model_type kmeans --num_cluster 6 --name_4D kmeans_4D_6_1x1 --name_3D kmeans_3D_6_1x1
python train.py --data_3D training_data_3D_1x1_0025 --data_4D training_data_4D_1x1_0025 --model_type kmeans --num_cluster 6 --name_4D kmeans_4D_6_1x1_0025 --name_3D kmeans_3D_6_1x1_0025

# mini-kmeans
# 2 clusters
python train.py --data_3D training_data_3D_1x1 --data_4D training_data_4D_1x1 --model_type mini_batch_kmeans --num_cluster 2 --name_4D mini_kmeans_4D_2_1x1 --name_3D mini_kmeans_3D_2_1x1
python train.py --data_3D training_data_3D_1x1_0025 --data_4D training_data_4D_1x1_0025 --model_type mini_batch_kmeans --num_cluster 2 --name_4D mini_kmeans_4D_2_1x1_0025 --name_3D mini_kmeans_3D_2_1x1_0025
# 3 clusters
python train.py --data_3D training_data_3D_1x1 --data_4D training_data_4D_1x1 --model_type mini_batch_kmeans --num_cluster 3 --name_4D mini_kmeans_4D_3_1x1 --name_3D mini_kmeans_3D_3_1x1
python train.py --data_3D training_data_3D_1x1_0025 --data_4D training_data_4D_1x1_0025 --model_type mini_batch_kmeans --num_cluster 3 --name_4D mini_kmeans_4D_3_1x1_0025 --name_3D mini_kmeans_3D_3_1x1_0025
# 4 clusters
python train.py --data_3D training_data_3D_1x1 --data_4D training_data_4D_1x1 --model_type mini_batch_kmeans --num_cluster 4 --name_4D mini_kmeans_4D_4_1x1 --name_3D mini_kmeans_3D_4_1x1
python train.py --data_3D training_data_3D_1x1_0025 --data_4D training_data_4D_1x1_0025 --model_type mini_batch_kmeans --num_cluster 4 --name_4D mini_kmeans_4D_4_1x1_0025 --name_3D mini_kmeans_3D_4_1x1_0025
# 5 clusters
python train.py --data_3D training_data_3D_1x1 --data_4D training_data_4D_1x1 --model_type mini_batch_kmeans --num_cluster 5 --name_4D mini_kmeans_4D_5_1x1 --name_3D mini_kmeans_3D_5_1x1
python train.py --data_3D training_data_3D_1x1_0025 --data_4D training_data_4D_1x1_0025 --model_type mini_batch_kmeans --num_cluster 5 --name_4D mini_kmeans_4D_5_1x1_0025 --name_3D mini_kmeans_3D_5_1x1_0025
# 6 clusters
python train.py --data_3D training_data_3D_1x1 --data_4D training_data_4D_1x1 --model_type mini_batch_kmeans --num_cluster 6 --name_4D mini_kmeans_4D_6_1x1 --name_3D mini_kmeans_3D_6_1x1
python train.py --data_3D training_data_3D_1x1_0025 --data_4D training_data_4D_1x1_0025 --model_type mini_batch_kmeans --num_cluster 6 --name_4D mini_kmeans_4D_6_1x1_0025 --name_3D mini_kmeans_3D_6_1x1_0025

# gmm
# 2 clusters
python train.py --data_3D training_data_3D_1x1 --data_4D training_data_4D_1x1 --model_type gmm --num_cluster 2 --name_4D gmm_4D_2_1x1 --name_3D gmm_3D_2_1x1
python train.py --data_3D training_data_3D_1x1_0025 --data_4D training_data_4D_1x1_0025 --model_type gmm --num_cluster 2 --name_4D gmm_4D_2_1x1_0025 --name_3D gmm_3D_2_1x1_0025
# 3 clusters
python train.py --data_3D training_data_3D_1x1 --data_4D training_data_4D_1x1 --model_type gmm --num_cluster 3 --name_4D gmm_4D_3_1x1 --name_3D gmm_3D_3_1x1
python train.py --data_3D training_data_3D_1x1_0025 --data_4D training_data_4D_1x1_0025 --model_type gmm --num_cluster 3 --name_4D gmm_4D_3_1x1_0025 --name_3D gmm_3D_3_1x1_0025
# 4 clusters
python train.py --data_3D training_data_3D_1x1 --data_4D training_data_4D_1x1 --model_type gmm --num_cluster 4 --name_4D gmm_4D_4_1x1 --name_3D gmm_3D_4_1x1
python train.py --data_3D training_data_3D_1x1_0025 --data_4D training_data_4D_1x1_0025 --model_type gmm --num_cluster 4 --name_4D gmm_4D_4_1x1_0025 --name_3D gmm_3D_4_1x1_0025
# 5 clusters
python train.py --data_3D training_data_3D_1x1 --data_4D training_data_4D_1x1 --model_type gmm --num_cluster 5 --name_4D gmm_4D_5_1x1 --name_3D gmm_3D_5_1x1
python train.py --data_3D training_data_3D_1x1_0025 --data_4D training_data_4D_1x1_0025 --model_type gmm --num_cluster 5 --name_4D gmm_4D_5_1x1_0025 --name_3D gmm_3D_5_1x1_0025
# 6 clusters
python train.py --data_3D training_data_3D_1x1 --data_4D training_data_4D_1x1 --model_type gmm --num_cluster 6 --name_4D gmm_4D_6_1x1 --name_3D gmm_3D_6_1x1
python train.py --data_3D training_data_3D_1x1_0025 --data_4D training_data_4D_1x1_0025 --model_type gmm --num_cluster 6 --name_4D gmm_4D_6_1x1_0025 --name_3D gmm_3D_6_1x1_0025



# Training 3x3x3 dataset on kmeans and mini-kmeans
# kmeans
# 2 clusters
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type kmeans --num_cluster 2 --name_4D kmeans_4D_2_3x3 --name_3D kmeans_3D_2_3x3
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type kmeans --num_cluster 2 --name_4D kmeans_4D_2_3x3_0025 --name_3D kmeans_3D_2_3x3_0025
# 3 clusters
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type kmeans --num_cluster 3 --name_4D kmeans_4D_3_3x3 --name_3D kmeans_3D_3_3x3
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type kmeans --num_cluster 3 --name_4D kmeans_4D_3_3x3_0025 --name_3D kmeans_3D_3_3x3_0025
# 4 clusters
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type kmeans --num_cluster 4 --name_4D kmeans_4D_4_3x3 --name_3D kmeans_3D_4_3x3
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type kmeans --num_cluster 4 --name_4D kmeans_4D_4_3x3_0025 --name_3D kmeans_3D_4_3x3_0025
# 5 clusters
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type kmeans --num_cluster 5 --name_4D kmeans_4D_5_3x3 --name_3D kmeans_3D_5_3x3
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type kmeans --num_cluster 5 --name_4D kmeans_4D_5_3x3_0025 --name_3D kmeans_3D_5_3x3_0025
# 6 clusters
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type kmeans --num_cluster 6 --name_4D kmeans_4D_6_3x3 --name_3D kmeans_3D_6_3x3
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type kmeans --num_cluster 6 --name_4D kmeans_4D_6_3x3_0025 --name_3D kmeans_3D_6_3x3_0025

# mini-kmeans
# 2 clusters
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type mini_batch_kmeans --num_cluster 2 --name_4D mini_kmeans_4D_2_3x3 --name_3D mini_kmeans_3D_2_3x3
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type mini_batch_kmeans --num_cluster 2 --name_4D mini_kmeans_4D_2_3x3_0025 --name_3D mini_kmeans_3D_2_3x3_0025
# 3 clusters
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type mini_batch_kmeans --num_cluster 3 --name_4D mini_kmeans_4D_3_3x3 --name_3D mini_kmeans_3D_3_3x3
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type mini_batch_kmeans --num_cluster 3 --name_4D mini_kmeans_4D_3_3x3_0025 --name_3D mini_kmeans_3D_3_3x3_0025
# 4 clusters
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type mini_batch_kmeans --num_cluster 4 --name_4D mini_kmeans_4D_4_3x3 --name_3D mini_kmeans_3D_4_3x3
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type mini_batch_kmeans --num_cluster 4 --name_4D mini_kmeans_4D_4_3x3_0025 --name_3D mini_kmeans_3D_4_3x3_0025
# 5 clusters
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type mini_batch_kmeans --num_cluster 5 --name_4D mini_kmeans_4D_5_3x3 --name_3D mini_kmeans_3D_5_3x3
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type mini_batch_kmeans --num_cluster 5 --name_4D mini_kmeans_4D_5_3x3_0025 --name_3D mini_kmeans_3D_5_3x3_0025
# 6 clusters
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type mini_batch_kmeans --num_cluster 6 --name_4D mini_kmeans_4D_6_3x3 --name_3D mini_kmeans_3D_6_3x3
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type mini_batch_kmeans --num_cluster 6 --name_4D mini_kmeans_4D_6_3x3_0025 --name_3D mini_kmeans_3D_6_3x3_0025

# gmm
# 2 clusters
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type gmm --num_cluster 2 --name_4D gmm_4D_2_3x3 --name_3D gmm_3D_2_3x3
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type gmm --num_cluster 2 --name_4D gmm_4D_2_3x3_0025 --name_3D gmm_3D_2_3x3_0025
# 3 clusters
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type gmm --num_cluster 3 --name_4D gmm_4D_3_3x3 --name_3D gmm_3D_3_3x3
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type gmm --num_cluster 3 --name_4D gmm_4D_3_3x3_0025 --name_3D gmm_3D_3_3x3_0025
# 4 clusters
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type gmm --num_cluster 4 --name_4D gmm_4D_4_3x3 --name_3D gmm_3D_4_3x3
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type gmm --num_cluster 4 --name_4D gmm_4D_4_3x3_0025 --name_3D gmm_3D_4_3x3_0025
# 5 clusters
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type gmm --num_cluster 5 --name_4D gmm_4D_5_3x3 --name_3D gmm_3D_5_3x3
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type gmm --num_cluster 5 --name_4D gmm_4D_5_3x3_0025 --name_3D gmm_3D_5_3x3_0025
# 6 clusters
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type gmm --num_cluster 6 --name_4D gmm_4D_6_3x3 --name_3D gmm_3D_6_3x3
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type gmm --num_cluster 6 --name_4D gmm_4D_6_3x3_0025 --name_3D gmm_3D_6_3x3_0025



# Training 5x5x5 dataset
# kmeans
# 2 clusters
python train.py --data_3D training_data_3D_5x5 --data_4D training_data_4D_5x5 --model_type kmeans --num_cluster 2 --name_4D kmeans_4D_2_5x5 --name_3D kmeans_3D_2_5x5
python train.py --data_3D training_data_3D_5x5_0025 --data_4D training_data_4D_5x5_0025 --model_type kmeans --num_cluster 2 --name_4D kmeans_4D_2_5x5_0025 --name_3D kmeans_3D_2_5x5_0025
# 3 clusters
python train.py --data_3D training_data_3D_5x5 --data_4D training_data_4D_5x5 --model_type kmeans --num_cluster 3 --name_4D kmeans_4D_3_5x5 --name_3D kmeans_3D_3_5x5
python train.py --data_3D training_data_3D_5x5_0025 --data_4D training_data_4D_5x5_0025 --model_type kmeans --num_cluster 3 --name_4D kmeans_4D_3_5x5_0025 --name_3D kmeans_3D_3_5x5_0025
# 4 clusters
python train.py --data_3D training_data_3D_5x5 --data_4D training_data_4D_5x5 --model_type kmeans --num_cluster 4 --name_4D kmeans_4D_4_5x5 --name_3D kmeans_3D_4_5x5
python train.py --data_3D training_data_3D_5x5_0025 --data_4D training_data_4D_5x5_0025 --model_type kmeans --num_cluster 4 --name_4D kmeans_4D_4_5x5_0025 --name_3D kmeans_3D_4_5x5_0025
# 5 clusters
python train.py --data_3D training_data_3D_5x5 --data_4D training_data_4D_5x5 --model_type kmeans --num_cluster 5 --name_4D kmeans_4D_5_5x5 --name_3D kmeans_3D_5_5x5
python train.py --data_3D training_data_3D_5x5_0025 --data_4D training_data_4D_5x5_0025 --model_type kmeans --num_cluster 5 --name_4D kmeans_4D_5_5x5_0025 --name_3D kmeans_3D_5_5x5_0025
# 6 clusters
python train.py --data_3D training_data_3D_5x5 --data_4D training_data_4D_5x5 --model_type kmeans --num_cluster 6 --name_4D kmeans_4D_6_5x5 --name_3D kmeans_3D_6_5x5
python train.py --data_3D training_data_3D_5x5_0025 --data_4D training_data_4D_5x5_0025 --model_type kmeans --num_cluster 6 --name_4D kmeans_4D_6_5x5_0025 --name_3D kmeans_3D_6_5x5_0025

# mini-kmeans
# 2 clusters
python train.py --data_3D training_data_3D_5x5 --data_4D training_data_4D_5x5 --model_type mini_batch_kmeans --num_cluster 2 --name_4D mini_kmeans_4D_2_5x5 --name_3D mini_kmeans_3D_2_5x5
python train.py --data_3D training_data_3D_5x5_0025 --data_4D training_data_4D_5x5_0025 --model_type mini_batch_kmeans --num_cluster 2 --name_4D mini_kmeans_4D_2_5x5_0025 --name_3D mini_kmeans_3D_2_5x5_0025
# 3 clusters
python train.py --data_3D training_data_3D_5x5 --data_4D training_data_4D_5x5 --model_type mini_batch_kmeans --num_cluster 3 --name_4D mini_kmeans_4D_3_5x5 --name_3D mini_kmeans_3D_3_5x5
python train.py --data_3D training_data_3D_5x5_0025 --data_4D training_data_4D_5x5_0025 --model_type mini_batch_kmeans --num_cluster 3 --name_4D mini_kmeans_4D_3_5x5_0025 --name_3D mini_kmeans_3D_3_5x5_0025
# 4 clusters
python train.py --data_3D training_data_3D_5x5 --data_4D training_data_4D_5x5 --model_type mini_batch_kmeans --num_cluster 4 --name_4D mini_kmeans_4D_4_5x5 --name_3D mini_kmeans_3D_4_5x5
python train.py --data_3D training_data_3D_5x5_0025 --data_4D training_data_4D_5x5_0025 --model_type mini_batch_kmeans --num_cluster 4 --name_4D mini_kmeans_4D_4_5x5_0025 --name_3D mini_kmeans_3D_4_5x5_0025
# 5 clusters
python train.py --data_3D training_data_3D_5x5 --data_4D training_data_4D_5x5 --model_type mini_batch_kmeans --num_cluster 5 --name_4D mini_kmeans_4D_5_5x5 --name_3D mini_kmeans_3D_5_5x5
python train.py --data_3D training_data_3D_5x5_0025 --data_4D training_data_4D_5x5_0025 --model_type mini_batch_kmeans --num_cluster 5 --name_4D mini_kmeans_4D_5_5x5_0025 --name_3D mini_kmeans_3D_5_5x5_0025
# 6 clusters
python train.py --data_3D training_data_3D_5x5 --data_4D training_data_4D_5x5 --model_type mini_batch_kmeans --num_cluster 6 --name_4D mini_kmeans_4D_6_5x5 --name_3D mini_kmeans_3D_6_5x5
python train.py --data_3D training_data_3D_5x5_0025 --data_4D training_data_4D_5x5_0025 --model_type mini_batch_kmeans --num_cluster 6 --name_4D mini_kmeans_4D_6_5x5_0025 --name_3D mini_kmeans_3D_6_5x5_0025

# gmm
# 2 clusters
python train.py --data_3D training_data_3D_5x5 --data_4D training_data_4D_5x5 --model_type gmm --num_cluster 2 --name_4D gmm_4D_2_5x5 --name_3D gmm_3D_2_5x5
python train.py --data_3D training_data_3D_5x5_0025 --data_4D training_data_4D_5x5_0025 --model_type gmm --num_cluster 2 --name_4D gmm_4D_2_5x5_0025 --name_3D gmm_3D_2_5x5_0025
# 3 clusters
python train.py --data_3D training_data_3D_5x5 --data_4D training_data_4D_5x5 --model_type gmm --num_cluster 3 --name_4D gmm_4D_3_5x5 --name_3D gmm_3D_3_5x5
python train.py --data_3D training_data_3D_5x5_0025 --data_4D training_data_4D_5x5_0025 --model_type gmm --num_cluster 3 --name_4D gmm_4D_3_5x5_0025 --name_3D gmm_3D_3_5x5_0025
# 4 clusters
python train.py --data_3D training_data_3D_5x5 --data_4D training_data_4D_5x5 --model_type gmm --num_cluster 4 --name_4D gmm_4D_4_5x5 --name_3D gmm_3D_4_5x5
python train.py --data_3D training_data_3D_5x5_0025 --data_4D training_data_4D_5x5_0025 --model_type gmm --num_cluster 4 --name_4D gmm_4D_4_5x5_0025 --name_3D gmm_3D_4_5x5_0025
# 5 clusters
python train.py --data_3D training_data_3D_5x5 --data_4D training_data_4D_5x5 --model_type gmm --num_cluster 5 --name_4D gmm_4D_5_5x5 --name_3D gmm_3D_5_5x5
python train.py --data_3D training_data_3D_5x5_0025 --data_4D training_data_4D_5x5_0025 --model_type gmm --num_cluster 5 --name_4D gmm_4D_5_5x5_0025 --name_3D gmm_3D_5_5x5_0025
# 6 clusters
python train.py --data_3D training_data_3D_5x5 --data_4D training_data_4D_5x5 --model_type gmm --num_cluster 6 --name_4D gmm_4D_6_5x5 --name_3D gmm_3D_6_5x5
python train.py --data_3D training_data_3D_5x5_0025 --data_4D training_data_4D_5x5_0025 --model_type gmm --num_cluster 6 --name_4D gmm_4D_6_5x5_0025 --name_3D gmm_3D_6_5x5_0025







# show single slice
python show_index.py --model_4D mini_kmeans_4D_3_3x3 --model_3D mini_kmeans_3D_3_3x3 --size 3 --timestamp 0025 --slice 601 --pore_4D 1,2 --pore_3D 2
python show_index.py --model_4D mini_kmeans_4D_3_3x3_0025 --model_3D mini_kmeans_3D_3_3x3_0025 --size 3 --timestamp 0025 --slice 601 --pore_4D 0 --pore_3D 1

python show_index.py --model_4D mini_kmeans_4D_3_1x1 --model_3D mini_kmeans_3D_3_1x1 --size 1 --timestamp 0025 --slice 601 --pore_4D 0 --pore_3D 1

python show_index.py --model_4D mini_kmeans_4D_3_5x5 --model_3D mini_kmeans_3D_3_5x5 --size 5 --timestamp 0025 --slice 601 --pore_4D 2 --pore_3D 2

python show_conv.py --model_4D mini_kmeans_4D_3_3x3 --model_3D mini_kmeans_3D_3_3x3 --timestamp 0025 --slice 601 --pore_4D 1,2 --pore_3D 2

# Validate data
python validate.py --model_4D mini_kmeans_4D_3_3x3 --model_3D mini_kmeans_3D_3_3x3 --filename_4D validation_data_4D_3 --filename_3D validation_data_3D_3 --pore_4D 1,2 --pore_3D 2
python validate.py --model_4D mini_kmeans_4D_3_3x3_0025 --model_3D mini_kmeans_3D_3_3x3_0025 --filename_4D validation_data_4D_3 --filename_3D validation_data_3D_3 --pore_4D 0 --pore_3D 1


# visualisation centre
python visualisation_cluster_centre.py --model_4D mini_kmeans_4D_2_3x3 --model_3D mini_kmeans_3D_2_3x3 --type mini_batch_kmeans
python visualisation_cluster_centre.py --model_4D mini_kmeans_4D_3_3x3 --model_3D mini_kmeans_3D_3_3x3 --type mini_batch_kmeans


# visualisation umap
python visualisation_umap.py






# Segment slices
python segment_index.py --model_4D mini_kmeans_4D_3_3x3 --model_3D mini_kmeans_3D_3_3x3 --size 3 --timestamp 0025 --pore_4D 2 --pore_3D 2

python segment_conv.py --model_4D mini_kmeans_4D_3_3x3 --model_3D mini_kmeans_3D_3_3x3 --timestamp 0025 --pore_4D 2 --pore_3D 2

python segment_conv.py --model_4D mini_kmeans_4D_3_3x3 --model_3D mini_kmeans_3D_3_3x3 --timestamp 0030 --pore_4D 2 --pore_3D 2




# upload data
gcloud compute scp --recurse ./SHP15_T113_0028 msc:~/MSc-Project/
# download data
gcloud compute scp --recurse msc:~/MSc-Project/SHP15_T113_0025/segmentation_3D ./results/
gcloud compute scp --recurse msc:~/MSc-Project/SHP15_T113_0025/segmentation_4D ./results/