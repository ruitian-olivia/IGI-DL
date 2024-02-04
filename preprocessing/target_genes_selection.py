import os
import csv
import gzip
import json
import scipy.io
import numpy as np
import pandas as pd

def text_save(content,filename,mode='w'):
    # Try to save a list variable in txt file.
    file = open(filename,mode)
    for i in range(len(content)):
        file.write(str(content[i])+'\n')
    file.close()

sample_list = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'sample6', 'sample7', 'sample8', 'sample9', 'sample10']

SVGs_Y_list = None
for sample_name in sample_list:
    SVGs_path = os.path.join('../preprocessed_data/SVG_top2000', sample_name+'_SPARKX.txt')
    with open(SVGs_path, "r", encoding="utf-8") as f:
        SVGs_1000_list = f.read().splitlines()
    if SVGs_Y_list is None:
        SVGs_Y_list = SVGs_1000_list
    else:
        SVGs_Y_list = list(set(SVGs_Y_list).intersection(set(SVGs_1000_list)))
    
print("len(SVGs_Y_list):", len(SVGs_Y_list))
SVGs_Y_list.sort()

ratio_all_df = None
for sample_name in sample_list:
    
    root_path = os.path.join("../dataset/", sample_name)
    
    matrix_dir = os.path.join(root_path, "filtered_feature_bc_matrix")
    mat = scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx.gz"))

    features_path = os.path.join(matrix_dir, "features.tsv.gz")
    gene_names = [row[1] for row in csv.reader(gzip.open(features_path, mode='rt'), delimiter="\t")]
    barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
    barcodes = [row[0] for row in csv.reader(gzip.open(barcodes_path, mode='rt'), delimiter="\t")]
    mat_df = pd.DataFrame(mat.toarray().T,index=barcodes, columns=gene_names)
        
    mat_df_driver = mat_df.loc[:, SVGs_Y_list]
    expression_ratio = mat_df_driver.astype(bool).sum(axis=0)/mat_df_driver.shape[0]
    expression_ratio_df = pd.DataFrame(expression_ratio)
    expression_ratio_df.columns = [sample_name]
    
    if ratio_all_df is None:
        ratio_all_df = expression_ratio_df
    else:
        ratio_all_df = pd.concat([ratio_all_df,expression_ratio_df],axis=1)
    print(ratio_all_df)
        
ratio_all_df = ratio_all_df.dropna(axis=0,how='any')
ratio_all_df["mean"] = ratio_all_df.mean(axis=1)
ratio_all_df = ratio_all_df.sort_values(by="mean",ascending=False)
print("=======================")
print(ratio_all_df)

SVGs_Y_filtered = ratio_all_df[(ratio_all_df['mean']>0.3)].index.tolist()
print("len(SVGs_Y_filtered):", len(SVGs_Y_filtered))
text_save(SVGs_Y_filtered, 'SVGs_SPARKX.txt')
