import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class patch_train_data(Dataset):

    def __init__(self, root, exclude_tissue, transform):
        super(patch_train_data, self).__init__()

        label_root_path = '../../preprocessed_data/y_label_df'
        self.root = root
        self.exclude_tissue = exclude_tissue
        self.transform = transform

        tissue_list = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5']

        img_list = []
        for tissue_id in tissue_list:
            if tissue_id != exclude_tissue:
                tissue_path = os.path.join(root,tissue_id)
                label_path = os.path.join(label_root_path,tissue_id,'log_norm_df.csv')
                label_df = pd.read_csv(label_path, index_col = 0)

                for img_file in os.listdir(tissue_path):
                    if img_file.endswith("png"):
                        img_p = os.path.join(tissue_path, img_file)
                        patch_name = img_file.split(".")[0]
                        y_np = np.array(label_df.loc[[patch_name]])
                        y_tensor = torch.squeeze(torch.tensor(y_np, dtype=torch.float))

                        img_list.append([img_p, y_tensor])

        self.imgs = img_list

    def __getitem__(self, index):
        img_path = self.imgs[index][0]
        y_tensor = self.imgs[index][1]

        pil_img = Image.open(img_path).convert("RGB")
        img_data = self.transform(pil_img)

        return img_data, y_tensor

    def __len__(self):
        return len(self.imgs)

class patch_test_data(Dataset):

    def __init__(self, root, test_tissue, transform):
        super(patch_test_data, self).__init__()

        label_root_path = '../../preprocessed_data/y_label_df'
        visium_root_dir = "../../dataset"
        self.root = root
        self.test_tissue = test_tissue
        self.transform = transform

        img_list = []
        tissue_path = os.path.join(root,test_tissue)
        label_path = os.path.join(label_root_path,test_tissue,'log_norm_df.csv')
        label_df = pd.read_csv(label_path, index_col = 0)

        visium_path = os.path.join(visium_root_dir,test_tissue)
        scalefactor_file = os.path.join(visium_path, "spatial/scalefactors_json.json")
        with open(scalefactor_file, 'r', encoding = 'utf-8') as f:
            scalefactor_dict = json.load(f)
        scalef = scalefactor_dict['tissue_hires_scalef']

        spot_list_file = os.path.join(visium_path,"spatial/tissue_positions_list.csv")
        spot_coord_df = pd.read_csv(spot_list_file,
                                header=None, names= ['barcodes','tissue','row','col','imgrow','imgcol'])

        spot_coord_df["hires_row"] = round(spot_coord_df['imgrow'] * scalef).astype(int)
        spot_coord_df["hires_col"] = round(spot_coord_df['imgcol'] * scalef).astype(int)
        spot_coord_tissue = spot_coord_df.loc[spot_coord_df.tissue==1,:].set_index('barcodes')

        for img_file in os.listdir(tissue_path):
            if img_file.endswith("png"):
                img_p = os.path.join(tissue_path, img_file)
                patch_name = img_file.split(".")[0]
                y_np = np.array(label_df.loc[[patch_name]])
                y_tensor = torch.squeeze(torch.tensor(y_np, dtype=torch.float))
                x_coor = np.array(spot_coord_tissue.loc[[patch_name]]['hires_row'])
                x_coor_tensor = torch.tensor(x_coor, dtype=torch.int)
                y_coor = np.array(spot_coord_tissue.loc[[patch_name]]['hires_col'])
                y_coor_tensor = torch.tensor(y_coor, dtype=torch.int)

                img_list.append([img_p, y_tensor, patch_name, x_coor_tensor, y_coor_tensor])

        self.imgs = img_list

    def __getitem__(self, index):
        img_path = self.imgs[index][0]
        y_tensor = self.imgs[index][1]
        patch_name = self.imgs[index][2]
        x_coor_tensor = self.imgs[index][3]
        y_coor_tensor = self.imgs[index][4]
        pil_img = Image.open(img_path).convert("RGB")
        img_data = self.transform(pil_img)

        return img_data, y_tensor, patch_name, x_coor_tensor, y_coor_tensor

    def __len__(self):
        return len(self.imgs)
