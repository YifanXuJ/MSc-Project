import os
import numpy as np
import pandas as pd

# 4 modes
PRECISION_50 = 1
PRECISION_X = 2
FSCORE_SOLO = 3
PRECISION_SOLO = 4

def auto_label(seg_model, seg_nd, cluster_num, mode, threshold=0.5):
    # cluster_num: the total number of clusters
    assert mode in [1,2,3,4], "Invalid mode: mode should be integer in [1,2,3,4]."
    if mode == 1:
        assert threshold == 0.5, "Mode 1 requires threshold = 0.5."

    csv_file = os.path.join(os.getcwd(), 'evaluation_rec_f1', '{}_{}_{}_f1.csv'.format(seg_model, seg_nd, cluster_num))
    df = pd.read_csv(csv_file, usecols=['slice', 'current_cluster','pore_micro_precision', 'pore_micro_f1', 'gypsum_micro_precision', 'gypsum_micro_f1', 'celestite_micro_precision', 'celestite_micro_f1', 'bassanite_micro_precision', 'bassanite_micro_f1'])
    
    label = [0]*cluster_num

    for i in range(cluster_num):
        one_cluster = df.loc[df['current_cluster'] == i]
        stats = one_cluster.mean()
        precisions = [stats[2], stats[4], stats[6], stats[8]]
        fscores = [stats[3], stats[5], stats[7], stats[9]]
        p_max = max(precisions)
        p_idx = np.argmax(precisions)
        f_idx = np.argmax(fscores)
        if mode in [1,2]:
            if p_max <= threshold:
                idx = f_idx
            else:
                idx = p_idx
        elif mode == 3:
            idx = f_idx
        else:
            idx = p_idx
            
        class_num = idx + 1
        label[i] = class_num

    return label


