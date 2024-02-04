import os
import math
import pandas as pd
import numpy as np

norm_root_path = '../../preprocessed_data/filtered_nuclei_standar_features'
tissue_list = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'sample6', 'sample7', 'sample8', 'sample9', 'sample10']

merged_feature_df = pd.DataFrame()

for tissue_name in tissue_list:
    print("Tissue name:", tissue_name)
    tissue_feature_df = pd.DataFrame()
    norm_feature_path = os.path.join(norm_root_path, tissue_name)
    
    for feature_file in os.listdir(norm_feature_path):
        if feature_file.endswith('.csv'):
            nuclei_feature_df = pd.read_csv(os.path.join(norm_feature_path, feature_file), index_col = 0)
            tissue_feature_df  = pd.concat([tissue_feature_df, nuclei_feature_df], axis=0)
            
    tissue_feature_df['slide.name'] = tissue_name 
    merged_feature_df = pd.concat([merged_feature_df,tissue_feature_df], axis=0)

print("merged_feature_df.shape:", merged_feature_df.shape)
merged_feature_df = merged_feature_df.drop(['type_0','type_1','type_2','type_3','type_4','type_5'], axis=1)
print("merged_feature_df.shape(after deleting type features):", merged_feature_df.shape)

nuclei_corr_matrix = merged_feature_df.corr()

threshold = 0.9
removed_features = []
for feature1 in nuclei_corr_matrix.columns:
    for feature2 in nuclei_corr_matrix.columns:
        if feature1 != feature2 and abs(nuclei_corr_matrix.loc[feature1, feature2]) > threshold:
            mean_abs_corr_feature1 = abs(nuclei_corr_matrix[feature1]).mean()
            mean_abs_corr_feature2 = abs(nuclei_corr_matrix[feature2]).mean()

            if mean_abs_corr_feature1 > mean_abs_corr_feature2:
                removed_features.append(feature1)
            else:
                removed_features.append(feature2)

removed_feature_list = list(set(removed_features))
print("len(removed_feature_list):", len(removed_feature_list))
with open("CRC_removed_feature_list.txt", "w") as file:
    for item in removed_feature_list:
        file.write(item + "\n")

