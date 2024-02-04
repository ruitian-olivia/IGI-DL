# Extract a set of numerical features for each nuclei using HistomicsTK package.
import os
import sys
import json
import math
import heapq
import shutil
import argparse
import skimage.io
import numpy as np
import pandas as pd
import histomicstk as htk
import matplotlib.pyplot as plt

def extract_seg_features(img_file_path, mask_dir_path, feature_path, barcode):
    """
    It is a function to obtain a set of numerical features for each nuclei.
    Arguments
        img_file_path: the file path of the input color normalized patch.
        mask_dir_path: the file path of segmentation masks predicted by Hover-Net.
        feature_path: the file path for saving extracted nuclei features in the patch.
        barcode: the barcode ID of the input patch.
    """
    im_input = skimage.io.imread(img_file_path)[:, :, :3] 
    # create stain to color map
    stainColorMap = {
        'hematoxylin': [0.65, 0.70, 0.29],
        'eosin':       [0.07, 0.99, 0.11],
        'dab':         [0.27, 0.57, 0.78],
        'null':        [0.0, 0.0, 0.0]
    }
    # specify stains of input image
    stain_1 = 'hematoxylin'   # nuclei stain
    stain_2 = 'eosin'         # cytoplasm stain
    stain_3 = 'null'          # set to null of input contains only two stains
    # create stain matrix
    W = np.array([stainColorMap[stain_1],
                  stainColorMap[stain_2],
                  stainColorMap[stain_3]]).T
    # perform standard color deconvolution
    im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_input, W).Stains
    
    npy_path = os.path.join(mask_dir_path, 'instances.npy')
    im_nuclei_seg_mask = np.load(npy_path)
    im_nuclei_stain = im_stains[:, :, 0]
    nuclei_features = htk.features.compute_nuclei_features(im_nuclei_seg_mask, im_nuclei_stain)
    nuclei_features["Label"] = nuclei_features["Label"].astype(int)
    
    nuclei_num = len(nuclei_features["Label"])
    if nuclei_num >= 5:
    
        centroid_np = np.array(nuclei_features[["Identifier.CentroidX", "Identifier.CentroidY"]])

        json_path = os.path.join(mask_dir_path, 'nuclei_dict.json')
        with open(json_path) as json_file:
            json_data = json.load(json_file)
        type_df = pd.DataFrame.from_dict(json_data, orient='index',columns=['type'])
        type_df = type_df.reset_index()
        type_df['Label'] = type_df.index.astype('int')
        type_df = type_df[['Label', 'type']]

        merge_df = pd.merge(type_df, nuclei_features, on="Label")
        merge_df = merge_df.drop(['Identifier.Xmin', 'Identifier.Ymin',\
                              'Identifier.Xmax', 'Identifier.Ymax',\
                              'Identifier.WeightedCentroidX','Identifier.WeightedCentroidY'],axis=1)
        merge_df.to_csv(os.path.join(feature_path, str(barcode)+'.csv'), header=True, index=False)
    
tissue_list = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'sample6', 'sample7', 'sample8', 'sample9', 'sample10']

mask_root_path = '../preprocessed_data/hover_seg'
img_root_path = '../preprocessed_data/HE_nmzd'
feature_root_path = '../preprocessed_data/nuclei_seg_features'

for tissue_name in tissue_list:
    img_path = os.path.join(img_root_path, tissue_name)
    mask_path = os.path.join(mask_root_path, tissue_name)
    feature_path = os.path.join(feature_root_path, tissue_name)
    
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    for filename in os.listdir(img_path):
        if filename.endswith('png'):
            try:
                barcode = filename[:-4]
                img_file_path = os.path.join(img_path, filename)
                mask_dir_path = os.path.join(mask_path, barcode)
                feature_dir_path = os.path.join(feature_path, barcode)

                extract_seg_features(img_file_path, mask_dir_path, feature_path, barcode)
                
            except:
                print("Error occured in %s" % os.path.join(img_path, filename))

