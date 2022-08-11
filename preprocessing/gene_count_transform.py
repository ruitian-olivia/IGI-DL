# Transform the raw gene counts data with several steps
import os
import csv
import gzip
import json
import scipy.io
import numpy as np
import pandas as pd

# Load selected target genes list
predict_gene_path = './predict_gene_list.txt'
with open(predict_gene_path, "r", encoding="utf-8") as f:
    target_y_list = f.read().splitlines()

data_root_dir = "../dataset"
save_root_path = '../preprocessed_data/y_label_df'

tissue_list = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5']

for tissue_name in tissue_list:
    
    root_path = os.path.join(data_root_dir, tissue_name)
    
    matrix_dir = os.path.join(root_path, "filtered_feature_bc_matrix")
    mat = scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx.gz"))

    features_path = os.path.join(matrix_dir, "features.tsv.gz")
    gene_names = [row[1] for row in csv.reader(gzip.open(features_path, mode='rt'), delimiter="\t")]
    barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
    barcodes = [row[0] for row in csv.reader(gzip.open(barcodes_path, mode='rt'), delimiter="\t")]
    mat_df = pd.DataFrame(mat.toarray().T,index=barcodes, columns=gene_names)
    
    save_count_path = os.path.join(save_root_path, tissue_name)
    if not os.path.exists(save_count_path):
        os.makedirs(save_count_path)
    
    UMI_row_sum = mat_df.sum(axis=1)
    N_count_df = pd.DataFrame(UMI_row_sum, columns=["Count_N"])
    print(tissue_name,"N_count_df.median():", N_count_df.median())
    print("N_count_df:",(np.nan in N_count_df))
    N_count_df.to_csv(os.path.join(save_count_path,'N_count_df.csv'), header=True)
    
    count_target_df = mat_df.loc[:, target_y_list]
    print("count_target_df:",count_target_df.isnull().values.any())
    count_target_df.to_csv(os.path.join(save_count_path,'count_target_df.csv'), header=True)
    
    # pseudo counts construction
    pseudo_count_df = count_target_df + 1
    # spot-level normalization & scaling
    transformed_count_df = pseudo_count_df.div((mat_df+1).sum(axis=1), axis='rows') * 1000000
    # log transformation
    log_norm_df = np.log(transformed_count_df)
    print("log_norm_df", log_norm_df.isnull().values.any())
    log_norm_df.to_csv(os.path.join(save_count_path,'log_norm_df.csv'), header=True, float_format='%.4f')

