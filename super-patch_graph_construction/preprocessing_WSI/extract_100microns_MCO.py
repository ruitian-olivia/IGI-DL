import os
import glob
import multiprocessing
import pandas as pd
import numpy as np
import openslide as osd
from skimage.filters import threshold_multiotsu

MCO_svs_path = './MCO-CRC-dataset'
MCO_svs_files = glob.glob(os.path.join(MCO_svs_path, "*.svs"))

print("len(MCO_svs_files):", len(MCO_svs_files))

def extract_100microns(image):
    microns_length = 100
    save_dir = './preprocessed_MCO/HE_patches/'
    
    print("Image path:", image)
    sample = image.split('/')[-1].split('.')[0]
    print("Sampe name:", sample)
    try:
        slideimage = osd.OpenSlide(image)
        print(slideimage.level_downsamples)
        print(slideimage.properties)
        mpp = slideimage.properties['aperio.MPP']
        print("aperio.MPP:", mpp)
        target_mpp = float(mpp)
    except:
        print('openslide error')
        print("Image path:", image)
        return 0

    downsampling = slideimage.level_downsamples
    if len(downsampling) > 2:
        save_path = os.path.join(save_dir, sample)
        if not os.path.exists(save_path):
            os.makedirs(save_path) 

        best_downsampling_level = 2
        downsampling_factor = int(slideimage.level_downsamples[best_downsampling_level])

        # Get the image at the requested scale
        svs_native_levelimg = slideimage.read_region((0, 0), best_downsampling_level, slideimage.level_dimensions[best_downsampling_level])
        svs_native_levelimg = svs_native_levelimg.convert('L')
        img = np.array(svs_native_levelimg)

        thresholds = threshold_multiotsu(img)
        print('thresholds:', thresholds)
        regions = np.digitize(img, bins=thresholds)
        regions[regions == 1] = 0
        regions[regions == 2] = 1
        thresh_otsu = regions

        imagesize = round(microns_length/target_mpp)
        print('imagesize:', imagesize)
        downsampled_size = int(round(imagesize /downsampling_factor))
        Width = slideimage.dimensions[0]
        Height = slideimage.dimensions[1]
        num_row = int(round(Height/imagesize)) + 1
        num_col = int(round(Width/imagesize)) + 1

        for i in range(0, num_col):
            for j in range(0, num_row):

                if thresh_otsu.shape[1] >= (i+1)*downsampled_size:
                    if thresh_otsu.shape[0] >= (j+1)*downsampled_size:
                        cut_thresh = thresh_otsu[j*downsampled_size:(j+1)*downsampled_size, i*downsampled_size:(i+1)*downsampled_size]
                    else:
                        cut_thresh = thresh_otsu[(j)*downsampled_size:thresh_otsu.shape[0], i*downsampled_size:(i+1)*downsampled_size]
                else:
                    if thresh_otsu.shape[0] >= (j+1)*downsampled_size:
                        cut_thresh = thresh_otsu[j*downsampled_size:(j+1)*downsampled_size, (i)*downsampled_size:thresh_otsu.shape[1]]
                    else:
                        cut_thresh = thresh_otsu[(j)*downsampled_size:thresh_otsu.shape[0], (i)*downsampled_size:thresh_otsu.shape[1]]

                if np.mean(cut_thresh) > 0.75:
                    pass
                else:
                    filter_location = (i*imagesize, j*imagesize)
                    level = 0
                    patch_size = (imagesize, imagesize)
                    location = (filter_location[0], filter_location[1])

                    CutImage = slideimage.read_region(location, level, patch_size)
                    CutImage.save(os.path.join(save_path, 'X_{}_Y_{}.png'.format(str(i), str(j))))


pool_obj = multiprocessing.Pool(64)
answer = pool_obj.map(extract_100microns, MCO_svs_files)

