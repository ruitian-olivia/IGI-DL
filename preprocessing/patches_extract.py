# Segment the whole tissue slide image into many patches according to the coordinates of the spot.
import os
import cv2
import glob
import json
import pandas as pd

sample_list = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'sample6', 'sample7', 'sample8', 'sample9', 'sample10']

for sample_name in sample_list:
    patch_save_dir = os.path.join("../preprocessed_data/HE_patches", sample_name)
    if not os.path.exists(patch_save_dir):
        os.makedirs(patch_save_dir)   

    file_root_path = os.path.join("../dataset/", sample_name)
    spot_list_file = os.path.join(file_root_path, "spatial/tissue_positions_list.csv")
    scalefactor_file = os.path.join(file_root_path, "spatial/scalefactors_json.json")
    HE_path = os.path.join(file_root_path, 'HE_image')

    if len(glob.glob(os.path.join(HE_path, "*.tif"))) != 0:
        tissue_img_file = glob.glob(os.path.join(HE_path, "*.tif"))[0]
    else:
        tissue_img_file = glob.glob(os.path.join(HE_path, "*.jpg"))[0]

    spot_coord = pd.read_csv(spot_list_file,
                            header=None, names= ['barcodes','tissue','row','col','imgrow','imgcol'])
    raw_img = cv2.imread(tissue_img_file,1)
    print('Input image dimension:', raw_img.shape)
    spot_coord_tissue = spot_coord.loc[spot_coord.tissue==1,:]

    with open(scalefactor_file, 'r', encoding = 'utf-8') as f:
        scalefactor_dict = json.load(f)

    fullres = scalefactor_dict['spot_diameter_fullres']

    sz = round(fullres*10/13) # From 10X: the original slide design has a 65microns spot diameter.

    for index, row in spot_coord_tissue.iterrows():
        row_center = round(row["imgrow"])
        col_center = round(row["imgcol"])
        spot_patch = raw_img[row_center-sz:row_center+sz, \
            col_center-sz:col_center+sz]
        
        cv2.imwrite(os.path.join(patch_save_dir, row["barcodes"]+".png"),
                    spot_patch, [cv2.IMWRITE_PNG_COMPRESSION, 0])