# Reinhard color normalization for HE-stained histological images using histomicsTK tools.
# Reference: https://digitalslidearchive.github.io/HistomicsTK/examples/nuclei_segmentation.html#Perform-color-normalization
import os
import PIL
import skimage.io
import skimage.color
import histomicstk as htk

def nmzd_reinhard_rescale(input_image_file, nmzd_path, barcode):
    """
    It is a function to perform Reinhard color normalization.
    Use the 'ref_HE.png' as a reference.
    Arguments
        input_image_file: the file path of the input patch.
        nmzd_path: the file path for saving normalized HE patches.
        barcode: the barcode ID of the input patch.
    """
    rescale_size = 200
    im_input = skimage.io.imread(input_image_file)[:, :, :3]
    # Load reference image for normalization
    ref_image_file = 'ref_HE.png' 
    im_reference = skimage.io.imread(ref_image_file)[:, :, :3]
    # get mean and stddev of reference image in lab space
    mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(im_reference)
    # perform reinhard color normalization
    im_nmzd = htk.preprocessing.color_normalization.reinhard(im_input, mean_ref, std_ref)
    pil_img = PIL.Image.fromarray(im_nmzd)
    pil_img = pil_img.resize(size=(rescale_size, rescale_size))
    pil_img.save(os.path.join(nmzd_path, barcode+".png"))

tissue_list = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'sample6', 'sample7', 'sample8', 'sample9', 'sample10']


for tissue_name in tissue_list:
    source_path = os.path.join("../preprocessed_data/HE_patches",tissue_name)
    save_root_path = "../preprocessed_data//HE_nmzd"
    nmzd_path = os.path.join(save_root_path, tissue_name)
    if not os.path.exists(nmzd_path):
        os.makedirs(nmzd_path)

    for filename in os.listdir(source_path):
        if filename.endswith('png'):
            try:
                barcode = filename[:-4]
                input_img_file = os.path.join(source_path, filename)
                nmzd_reinhard_rescale(input_img_file, nmzd_path, barcode)

            except:
                print("Error occured in %s" % os.path.join(source_path, filename))
    
    print("End of normalization & rescaling of %s" % tissue_name)