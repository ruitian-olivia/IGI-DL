import os
import sys
import json
import numpy as np
import pandas as pd

def extract_edge_ids(im_nuclei_seg_mask, pixel_num):
    top_row_coords = np.argwhere(im_nuclei_seg_mask[0, :] != 0)
    bottom_row_coords = np.argwhere(im_nuclei_seg_mask[-1, :] != 0)

    top_row_coords = np.column_stack((np.zeros_like(top_row_coords), top_row_coords))
    bottom_row_coords = np.column_stack((np.full_like(bottom_row_coords, im_nuclei_seg_mask.shape[0]-1), bottom_row_coords))

    left_column_coords = np.argwhere(im_nuclei_seg_mask[:, 0] != 0)
    right_column_coords = np.argwhere(im_nuclei_seg_mask[:, -1] != 0)

    left_column_coords = np.column_stack((left_column_coords, np.zeros_like(left_column_coords)))
    right_column_coords = np.column_stack((right_column_coords, np.full_like(right_column_coords, im_nuclei_seg_mask.shape[1]-1)))

    edge_coords = np.vstack((top_row_coords, bottom_row_coords, left_column_coords, right_column_coords))
    unique_edge_coords = np.unique(edge_coords, axis=0)
    
    nuclei_ids = im_nuclei_seg_mask[unique_edge_coords[:, 0], unique_edge_coords[:, 1]] 
    
    unique_ids, counts = np.unique(nuclei_ids, return_counts=True)

    edge_ids = []
    for i, nuclei_id in enumerate(unique_ids):
        if nuclei_id != 0:  
            if counts[i] >= pixel_num:
                edge_ids.append(nuclei_id)
                
    return edge_ids

tissue_list = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'sample6', 'sample7', 'sample8', 'sample9', 'sample10']
pixel_num = 10

mask_root_path = '../preprocessed_data/hover_seg'
feature_root_path = '../preprocessed_data/nuclei_seg_features'
filtered_feature_root_path = '../preprocessed_data/filtered_nuclei_seg_features'

for tissue_name in tissue_list:
    mask_path = os.path.join(mask_root_path, tissue_name)
    feature_path = os.path.join(feature_root_path, tissue_name)
    filtered_feature_path = os.path.join(filtered_feature_root_path, tissue_name)
    
    if not os.path.exists(filtered_feature_path):
        os.makedirs(filtered_feature_path)
        
    edge_num_dict = {}

    for filename in os.listdir(feature_path):
        if filename.endswith('csv'):
            barcode = filename[:-4]
            feature_csv_path = os.path.join(feature_path, filename)
            
            feature_df = pd.read_csv(feature_csv_path)

            mask_dir_path = os.path.join(mask_path, barcode)
            npy_path = os.path.join(mask_dir_path, 'instances.npy')
            im_nuclei_seg_mask = np.load(npy_path)
            edge_ids = extract_edge_ids(im_nuclei_seg_mask, pixel_num)
            edge_num_dict[barcode] = len(edge_ids)
            
            feature_df_filtered = feature_df[~feature_df['Label'].isin(edge_ids)]
            feature_df_filtered.to_csv(os.path.join(filtered_feature_path, str(barcode)+'.csv'), header=True, index=False)
            
    edge_num_df = pd.DataFrame([edge_num_dict]).T
    edge_num_df.to_csv(os.path.join(filtered_feature_root_path, str(tissue_name)+'_edge_num.csv'), header=False)

            



    