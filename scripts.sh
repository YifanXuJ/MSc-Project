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



# Training command
# Different number of clusters
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type mini_batch_kmeans --num_cluster 2 --name_4D mini_kmeans_4D_2_3x3_0025 --name_3D mini_kmeans_3D_2_3x3_0025
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type mini_batch_kmeans --num_cluster 3 --name_4D mini_kmeans_4D_3_3x3_0025 --name_3D mini_kmeans_3D_3_3x3_0025
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type mini_batch_kmeans --num_cluster 4 --name_4D mini_kmeans_4D_4_3x3_0025 --name_3D mini_kmeans_3D_4_3x3_0025
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type mini_batch_kmeans --num_cluster 5 --name_4D mini_kmeans_4D_5_3x3_0025 --name_3D mini_kmeans_3D_5_3x3_0025
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type mini_batch_kmeans --num_cluster 6 --name_4D mini_kmeans_4D_6_3x3_0025 --name_3D mini_kmeans_3D_6_3x3_0025

# Different feature size
python train.py --data_3D training_data_3D_1x1_0025 --data_4D training_data_4D_1x1_0025 --model_type mini_batch_kmeans --num_cluster 3 --name_4D mini_kmeans_4D_3_1x1_0025 --name_3D mini_kmeans_3D_3_1x1_0025
python train.py --data_3D training_data_3D_5x5_0025 --data_4D training_data_4D_5x5_0025 --model_type mini_batch_kmeans --num_cluster 3 --name_4D mini_kmeans_4D_3_5x5_0025 --name_3D mini_kmeans_3D_3_5x5_0025

# Different algorithms
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type kmeans --num_cluster 3 --name_4D kmeans_4D_3_3x3_0025 --name_3D kmeans_3D_3_3x3_0025
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type gmm --num_cluster 3 --name_4D gmmfull_4D_3_3x3_0025 --name_3D gmmfull_3D_3_3x3_0025 --covariance_type full
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type gmm --num_cluster 3 --name_4D gmmtied_4D_3_3x3_0025 --name_3D gmmtied_3D_3_3x3_0025 --covariance_type tied
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type gmm --num_cluster 3 --name_4D gmmdiag_4D_3_3x3_0025 --name_3D gmmdiag_3D_3_3x3_0025 --covariance_type diag
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type gmm --num_cluster 3 --name_4D gmmsphe_4D_3_3x3_0025 --name_3D gmmsphe_3D_3_3x3_0025 --covariance_type spherical
python train.py --data_3D training_data_3D_3x3_0025 --data_4D training_data_4D_3x3_0025 --model_type mean_shift --name_4D meanshift_4D_3_3x3_0025 --name_3D meanshift_3D_3_3x3_0025

# Trained on all time stamp
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type mini_batch_kmeans --num_cluster 2 --name_4D mini_kmeans_4D_2_3x3 --name_3D mini_kmeans_3D_2_3x3
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type mini_batch_kmeans --num_cluster 3 --name_4D mini_kmeans_4D_3_3x3 --name_3D mini_kmeans_3D_3_3x3
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type mini_batch_kmeans --num_cluster 4 --name_4D mini_kmeans_4D_4_3x3 --name_3D mini_kmeans_3D_4_3x3
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type mini_batch_kmeans --num_cluster 5 --name_4D mini_kmeans_4D_5_3x3 --name_3D mini_kmeans_3D_5_3x3
python train.py --data_3D training_data_3D_3x3 --data_4D training_data_4D_3x3 --model_type mini_batch_kmeans --num_cluster 6 --name_4D mini_kmeans_4D_6_3x3 --name_3D mini_kmeans_3D_6_3x3



# Validate data
# validate data trained on 0025
python validate.py --model_4D mini_kmeans_4D_2_3x3_0025 --model_3D mini_kmeans_3D_2_3x3_0025 --filename_4D validation_data_4D_3x3 --filename_3D validation_data_3D_3x3 --pore_4D 1 --pore_3D 1
python validate.py --model_4D mini_kmeans_4D_3_3x3_0025 --model_3D mini_kmeans_3D_3_3x3_0025 --filename_4D validation_data_4D_3x3 --filename_3D validation_data_3D_3x3 --pore_4D 0 --pore_3D 0
python validate.py --model_4D mini_kmeans_4D_4_3x3_0025 --model_3D mini_kmeans_3D_4_3x3_0025 --filename_4D validation_data_4D_3x3 --filename_3D validation_data_3D_3x3 --pore_4D 3 --pore_3D 2
python validate.py --model_4D mini_kmeans_4D_5_3x3_0025 --model_3D mini_kmeans_3D_5_3x3_0025 --filename_4D validation_data_4D_3x3 --filename_3D validation_data_3D_3x3 --pore_4D 2,4 --pore_3D 2,3
python validate.py --model_4D mini_kmeans_4D_6_3x3_0025 --model_3D mini_kmeans_3D_6_3x3_0025 --filename_4D validation_data_4D_3x3 --filename_3D validation_data_3D_3x3 --pore_4D 2,5 --pore_3D 2,5

# validate data trained on all time stamp
python validate.py --model_4D mini_kmeans_4D_2_3x3 --model_3D mini_kmeans_3D_2_3x3 --filename_4D validation_data_4D_3x3 --filename_3D validation_data_3D_3x3 --pore_4D 1 --pore_3D 0
python validate.py --model_4D mini_kmeans_4D_3_3x3 --model_3D mini_kmeans_3D_3_3x3 --filename_4D validation_data_4D_3x3 --filename_3D validation_data_3D_3x3 --pore_4D 0,2 --pore_3D 2
python validate.py --model_4D mini_kmeans_4D_4_3x3 --model_3D mini_kmeans_3D_4_3x3 --filename_4D validation_data_4D_3x3 --filename_3D validation_data_3D_3x3 --pore_4D 0,2,3 --pore_3D 0
python validate.py --model_4D mini_kmeans_4D_5_3x3 --model_3D mini_kmeans_3D_5_3x3 --filename_4D validation_data_4D_3x3 --filename_3D validation_data_3D_3x3 --pore_4D 3 --pore_3D 5
python validate.py --model_4D mini_kmeans_4D_6_3x3 --model_3D mini_kmeans_3D_6_3x3 --filename_4D validation_data_4D_3x3 --filename_3D validation_data_3D_3x3 --pore_4D 1,3 --pore_3D 2,4

# validate for different feature size
python validate.py --model_4D mini_kmeans_4D_3_3x3_0025 --model_3D mini_kmeans_3D_3_3x3_0025 --filename_4D validation_data_4D_3x3 --filename_3D validation_data_3D_3x3 --pore_4D 0 --pore_3D 0
python validate.py --model_4D mini_kmeans_4D_3_1x1_0025 --model_3D mini_kmeans_3D_3_1x1_0025 --filename_4D validation_data_4D_1x1 --filename_3D validation_data_3D_1x1 --pore_4D 2 --pore_3D 0
python validate.py --model_4D mini_kmeans_4D_3_5x5_0025 --model_3D mini_kmeans_3D_3_5x5_0025 --filename_4D validation_data_4D_5x5 --filename_3D validation_data_3D_5x5 --pore_4D 2 --pore_3D 2

# validate for different algorithms
python validate.py --model_4D kmeans_4D_3_3x3_0025 --model_3D kmeans_3D_3_3x3_0025 --filename_4D validation_data_4D_3x3 --filename_3D validation_data_3D_3x3 --pore_4D 0 --pore_3D 1
python validate.py --model_4D mini_kmeans_4D_3_3x3_0025 --model_3D mini_kmeans_3D_3_3x3_0025 --filename_4D validation_data_4D_3x3 --filename_3D validation_data_3D_3x3 --pore_4D 0 --pore_3D 0
python validate.py --model_4D gmmfull_4D_3_3x3_0025 --model_3D gmmfull_3D_3_3x3_0025 --filename_4D validation_data_4D_3x3 --filename_3D validation_data_3D_3x3 --pore_4D 2 --pore_3D 1
python validate.py --model_4D gmmtied_4D_3_3x3_0025 --model_3D gmmtied_3D_3_3x3_0025 --filename_4D validation_data_4D_3x3 --filename_3D validation_data_3D_3x3 --pore_4D 0 --pore_3D 1
python validate.py --model_4D gmmdiag_4D_3_3x3_0025 --model_3D gmmdiag_3D_3_3x3_0025 --filename_4D validation_data_4D_3x3 --filename_3D validation_data_3D_3x3 --pore_4D 2 --pore_3D 0
python validate.py --model_4D gmmsphe_4D_3_3x3_0025 --model_3D gmmsphe_3D_3_3x3_0025 --filename_4D validation_data_4D_3x3 --filename_3D validation_data_3D_3x3 --pore_4D 1 --pore_3D 2
python validate.py --model_4D meanshift_4D_3_3x3_0025 --model_3D meanshift_3D_3_3x3_0025 --filename_4D validation_data_4D_3x3 --filename_3D validation_data_3D_3x3 --pore_4D  1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374 --pore_3D 1,2,3,4,5,6,7,8,9,10,11,12,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66



# visualisation centre
python visualisation_cluster_centre.py --model_4D mini_kmeans_4D_3_3x3_0025 --model_3D mini_kmeans_3D_3_3x3_0025 --type mini_batch_kmeans
# visualisation pore and non-pore
python visualisation_points.py
# visualisation umap
python visualisation_umap.py





# Segment slices
python segment_index.py --model_4D mini_kmeans_4D_3_3x3 --model_3D mini_kmeans_3D_3_3x3 --size 3 --timestamp 0025 --pore_4D 2 --pore_3D 2
python segment_conv.py --model_4D mini_kmeans_4D_3_3x3 --model_3D mini_kmeans_3D_3_3x3 --timestamp 0025 --pore_4D 2 --pore_3D 2
python segment_conv.py --model_4D mini_kmeans_4D_4_3x3 --model_3D mini_kmeans_3D_4_3x3 --timestamp 0030 --pore_4D 2 --pore_3D 3



# show single slice
python show_index.py --model_4D mini_kmeans_4D_3_3x3 --model_3D mini_kmeans_3D_3_3x3 --size 3 --timestamp 0025 --slice 601 --pore_4D 1,2 --pore_3D 2
python show_index.py --model_4D mini_kmeans_4D_3_3x3_0025 --model_3D mini_kmeans_3D_3_3x3_0025 --size 3 --timestamp 0025 --slice 601 --pore_4D 0 --pore_3D 1
python show_index.py --model_4D mini_kmeans_4D_3_1x1 --model_3D mini_kmeans_3D_3_1x1 --size 1 --timestamp 0025 --slice 601 --pore_4D 0 --pore_3D 1
python show_index.py --model_4D mini_kmeans_4D_3_5x5 --model_3D mini_kmeans_3D_3_5x5 --size 5 --timestamp 0025 --slice 601 --pore_4D 2 --pore_3D 2
python show_conv.py --model_4D mini_kmeans_4D_3_3x3 --model_3D mini_kmeans_3D_3_3x3 --timestamp 0025 --slice 601 --pore_4D 1,2 --pore_3D 2
python show_index.py --model_4D mini_kmeans_4D_4_3x3 --model_3D mini_kmeans_3D_4_3x3 --size 3 --timestamp 0030 --slice 300 --pore_4D 2 --pore_3D 2



# upload data
gcloud compute scp --recurse ./SHP15_T113_0028 msc:~/MSc-Project/
# download data
gcloud compute scp --recurse msc:~/MSc-Project/SHP15_T113_0025/segmentation_3D ./results/
gcloud compute scp --recurse msc:~/MSc-Project/SHP15_T113_0025/segmentation_4D ./results/