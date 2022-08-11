# Standardize numerical features of nucleus at the patient level
import os
import math
import argparse
import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

feature_root_path = '../preprocessed_data/nuclei_seg_features'
norm_root_path = '../preprocessed_data/nuclei_standar_features'

tissue_list = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5']

for tissue_name in tissue_list:
    feature_path = os.path.join(feature_root_path,tissue_name)
    norm_feature_path = os.path.join(norm_root_path,tissue_name)
    
    if not os.path.exists(norm_feature_path):
        os.makedirs(norm_feature_path)
    
    tissue_feature_df = None
    for feature_file in os.listdir(feature_path):
        if feature_file.endswith('.csv'):
            patch_name = feature_file.split('.')[0]
            print(patch_name)

            nuclei_feature_path = os.path.join(feature_path, feature_file)
            nuclei_feature_df = pd.read_csv(nuclei_feature_path, index_col = 0)
            nuclei_feature_df.insert(0, 'barcode', patch_name)
            
            if tissue_feature_df is None:
                tissue_feature_df = nuclei_feature_df
            else:
                tissue_feature_df  =pd.concat([tissue_feature_df,nuclei_feature_df], axis=0)
                        
    tissue_feature_df['type'] = tissue_feature_df['type'].apply(str)
    categorical_features = ['type']
    tissue_feature_df_cat = pd.get_dummies(tissue_feature_df[categorical_features])
    tissue_feature_df = tissue_feature_df.drop(categorical_features, axis=1)
    tissue_feature_df = pd.concat([tissue_feature_df_cat, tissue_feature_df], axis=1)
    
    df_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    ss = StandardScaler()
    
    scale_features = tissue_feature_df.columns[9:]
    tissue_feature_df[scale_features] = df_mean.fit_transform(tissue_feature_df[scale_features])
    tissue_feature_df[scale_features] = ss.fit_transform(tissue_feature_df[scale_features])
    
    barcode_groups = tissue_feature_df.groupby(tissue_feature_df.barcode)
    for feature_file in os.listdir(feature_path):
        if feature_file.endswith('.csv'):
            patch_name = feature_file.split('.')[0]
            normalized_path = os.path.join(norm_feature_path,feature_file)
            normalized_df = barcode_groups.get_group(patch_name).copy()
            normalized_df.drop(columns='barcode', inplace=True)
            if tissue_feature_df.isnull().any().any():
                print("tissue_feature_df.isnull().any().any()--path:", normalized_path)

            normalized_df.to_csv(normalized_path, header=True, float_format='%.3f')            