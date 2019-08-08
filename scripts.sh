
# For 3x3x3 dataset

# 2 clusters
python training.py --training_data_3D training_data_3D_3.npy --training_data_4D training_data_4D_3.npy --model_type kmeans --num_cluster 2 --model_4D kmeans_4D_2_3x3.model --model_3D kmeans_3D_2_3x3.model

python training.py --training_data_3D training_data_3D_3.npy --training_data_4D training_data_4D_3.npy --model_type mini_batch_kmeans --num_cluster 2 --model_4D mini_kmeans_4D_2_3x3.model --model_3D mini_kmeans_3D_2_3x3.model
python training.py --training_data_3D training_data_3D_3x3_10.npy --training_data_4D training_data_4D_3x3_10.npy --model_type mini_batch_kmeans --num_cluster 2 --model_4D mini_kmeans_4D_2_3x3_10.model --model_3D mini_kmeans_3D_2_3x3_10.model

# 4 clusters --> gmm
python training.py --training_data_3D training_data_3D_3.npy --training_data_4D training_data_4D_3.npy --model_type gmm --num_cluster 4 --model_4D gmm_4D_4_3x3.model --model_3D gmm_3D_4_3x3.model

# 4 clusters --> mini_batch_kmeans
python training.py --training_data_3D training_data_3D_3.npy --training_data_4D training_data_4D_3.npy --model_type mini_batch_kmeans --num_cluster 4 --model_4D mini_kmeans_4D_4_3x3.model --model_3D mini_kmeans_3D_4_3x3.model

# show the results
python show.py --model_4D kmeans_4D_2_3x3.model --model_3D kmeans_3D_2_3x3.model --size 3 --timestamp 5 --slice 700
python show.py --model_4D mini_kmeans_4D_2_3x3.model --model_3D mini_kmeans_3D_2_3x3.model --size 3 --timestamp 5 --slice 700
python show.py --model_4D gmm_4D_4_3x3.model --model_3D gmm_3D_4_3x3.model --size 3 --timestamp 5 --slice 700
python show.py --model_4D mini_kmeans_4D_4_3x3.model --model_3D mini_kmeans_3D_4_3x3.model --size 3 --timestamp 5 --slice 700

python show.py --model_4D mini_kmeans_4D_2_3x3_10.model --model_3D mini_kmeans_3D_2_3x3_10.model --size 3 --timestamp 5 --slice 700


# For 1x1x1 dataset
# 2 clusters
python training.py --training_data_3D training_data_3D_1.npy --training_data_4D training_data_4D_1.npy --model_type mini_batch_kmeans --num_cluster 2 --model_4D mini_kmeans_4D_2_1x1.model --model_3D mini_kmeans_3D_2_1x1.model

# show the results
python show.py --model_4D mini_kmeans_4D_2_1x1.model --model_3D mini_kmeans_3D_2_1x1.model --size 1 --timestamp 5 --slice 700





# For 5x5x5 dataset
# 2 clusters
python training.py --training_data_3D training_data_3D_5.npy --training_data_4D training_data_4D_5.npy --model_type mini_batch_kmeans --num_cluster 2 --model_4D mini_kmeans_4D_2_5x5.model --model_3D mini_kmeans_3D_2_5x5.model

# show the results
python show.py --model_4D mini_kmeans_4D_2_5x5.model --model_3D mini_kmeans_3D_2_5x5.model --size 5 --timestamp 5 --slice 700


python segment.py --model_4D mini_kmeans_4D_2_3x3.model --model_3D mini_kmeans_3D_2_3x3.model --size 3 --timestamp 5


