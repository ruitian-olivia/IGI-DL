import os
import cv2
import PIL
import math
import argparse
import skimage.io
import skimage.color
import multiprocessing
import histomicstk as htk
import pandas as pd

import wsi_tile_cleanup as cleanup

source_root_path = './preprocessed_TCGA/HE_patches/READ'
patient_list = os.listdir(source_root_path)

def norm_100microns(tissue_name):
    source_root_path = './preprocessed_TCGA/HE_patches/READ'
    norm_root_path = './preprocessed_TCGA/HE_nmzd/READ'

    rescale_size = 200
    # Load reference image for normalization
    ref_image_file = '../preprocessing/ref_HE.png' 
    im_reference = skimage.io.imread(ref_image_file)[:, :, :3]
    # get mean and stddev of reference image in lab space
    mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(im_reference)
    
    try:
        source_path = os.path.join(source_root_path,tissue_name)
        normalized_path = os.path.join(norm_root_path,tissue_name)
        if not os.path.exists(normalized_path):
            os.makedirs(normalized_path)  

        for filename in os.listdir(source_path):
            if filename.endswith('png'):
                try:
                    tile_path = os.path.join(source_path, filename)

                    vi_tile = cleanup.utils.read_image(tile_path)
                    bands = cleanup.utils.split_rgb(vi_tile)
                    colors = ["red", "green", "blue"]

                    perc_list = []

                    for color in colors:
                        perc = cleanup.filters.pen_percent(bands, color)
                        print(f"{color}: {perc*100:.3f}%")
                        perc_list.append(perc)

                    perc = cleanup.filters.blackish_percent(bands)
                    print(f"blackish: {perc*100:.3f}%")
                    perc_list.append(perc)

                    perc = cleanup.filters.bg_percent(bands)
                    print(f"background: {perc*100:.3f}%")
                    perc_list.append(perc)

                    if max(perc_list) < 0.4:
                        im_input = skimage.io.imread(tile_path)[:, :, :3]

                        # perform reinhard color normalization
                        im_nmzd = htk.preprocessing.color_normalization.reinhard(im_input, mean_ref, std_ref)
                        pil_img = PIL.Image.fromarray(im_nmzd)
                        pil_img = pil_img.resize(size=(rescale_size, rescale_size))
                        pil_img.save(os.path.join(normalized_path, filename))

                except:
                    print("Error occured in patch: %s" % os.path.join(source_path, filename))

    except:
        print("Error occured in patient_id: %s" % source_path)
        

pool_obj = multiprocessing.Pool(64)
answer = pool_obj.map(norm_100microns, patient_list)