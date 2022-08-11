# Construct a Nuclei-Graph for each HE patch
import os
import math
import json
import torch
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch_geometric.data import Data

patch_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) 
])

# save graph into .pt files
# critical distance 20 \mu m, pix width 0.5 \mu m
critical = 20/0.5

tissue_list = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5']

visium_root_dir = '../dataset'
patch_root_path = "../preprocessed_data/HE_nmzd"
feature_root_path = '../preprocessed_data/nuclei_standar_features'
label_root_path = '../preprocessed_data/y_label_df'
graph_img_root_path = '../preprocessed_data/graph_image_pt'

for tissue_name in tissue_list:
    visium_path = os.path.join(visium_root_dir,tissue_name)
    patch_path = os.path.join(patch_root_path,tissue_name)
    scalefactor_file = os.path.join(visium_path, "spatial/scalefactors_json.json")
    with open(scalefactor_file, 'r', encoding = 'utf-8') as f:
        scalefactor_dict = json.load(f)

    fullres = scalefactor_dict['spot_diameter_fullres']
    scalef = scalefactor_dict['tissue_hires_scalef']
    spot_radius = round(fullres*scalef/2)

    spot_list_file = os.path.join(visium_path,"spatial/tissue_positions_list.csv")
    spot_coord_df = pd.read_csv(spot_list_file,
                            header=None, names= ['barcodes','tissue','row','col','imgrow','imgcol'])

    spot_coord_df["hires_row"] = round(spot_coord_df['imgrow'] * scalef).astype(int)
    spot_coord_df["hires_col"] = round(spot_coord_df['imgcol'] * scalef).astype(int)

    spot_coord_tissue = spot_coord_df.loc[spot_coord_df.tissue==1,:].set_index('barcodes')

    feature_path = os.path.join(feature_root_path,tissue_name)
    label_path = os.path.join(label_root_path,tissue_name)

    log_norm_path = os.path.join(label_path,'log_norm_df.csv')
    log_norm_df = pd.read_csv(log_norm_path, index_col = 0)
    
    N_count_path = os.path.join(label_path,'N_count_df.csv')
    N_count_df = pd.read_csv(N_count_path, index_col = 0)
    
    count_target_path = os.path.join(label_path,'count_target_df.csv')
    count_target_df = pd.read_csv(count_target_path, index_col = 0)

    for file_name in os.listdir(feature_path):
        if file_name.endswith('.csv'):
            patch_name = file_name.split('.')[0]
            node_feature_path = os.path.join(feature_path,file_name)
            node_feature_df = pd.read_csv(node_feature_path)
            
            xc = np.array(node_feature_df['Identifier.CentroidX'])
            yc = np.array(node_feature_df['Identifier.CentroidY'])
            num_cell = len(xc)
            
            A = np.zeros([num_cell,num_cell])
            edge_coor = list()
            edge_index =list()
            edge_attr = list()
            for k in range(num_cell):
                for j in range(k+1,num_cell):
                    dist = np.sqrt((xc[k]-xc[j])**2+(yc[k]-yc[j])**2)
                    if dist<critical:
                        A[k,j] = critical/dist
                        A[j,k] = critical/dist
                        edge_coor_temp = np.array([xc[k],yc[k],xc[j],yc[j]],dtype=np.float64)
                        edge_coor.append(edge_coor_temp)
                        edge_coor_temp = np.array([xc[j],yc[j],xc[k],yc[k]],dtype=np.float64)
                        edge_coor.append(edge_coor_temp)
                        edge_index_temp = np.array([k,j],dtype=np.int)
                        edge_index.append(edge_index_temp)
                        edge_index_temp = np.array([j,k],dtype=np.int)
                        edge_index.append(edge_index_temp)
                        edge_attr_temp = np.array([A[k,j],A[j,k]],dtype=np.float)
                        edge_attr.append(edge_attr_temp)

            edge_coor_np = np.reshape(np.array(edge_coor),[len(edge_coor),4])
            edge_index_np = np.reshape(np.array(edge_index),[len(edge_index),2])
            edge_attr_np = np.reshape(np.array(edge_attr),[len(edge_index),1])

            edge_index_tensor = torch.tensor(edge_index_np, dtype=torch.long)
            edge_attr_tensor = torch.tensor(edge_attr_np, dtype=torch.float)

            feature_np = np.array(node_feature_df.drop(['Label','Identifier.CentroidX', 'Identifier.CentroidY'],axis=1))
            feature_tensor = torch.tensor(feature_np, dtype=torch.float)
            
            if torch.isinf(feature_tensor).any():
                print("torch.isinf(feature_tensor).any()--path:", node_feature_path)
            if torch.isnan(feature_tensor).any():
                print("torch.isnan(feature_tensor).any()--path:", node_feature_path)
            
            pos_np = np.array(node_feature_df[['Identifier.CentroidX', 'Identifier.CentroidY']])
            pos_tensor = torch.tensor(pos_np, dtype=torch.float)
            
            y_log_norm = np.array(log_norm_df.loc[[patch_name]])
            y_log_tensor = torch.tensor(y_log_norm, dtype=torch.float)
            if torch.isinf(y_log_tensor).any():
                print("torch.isinf(y_log_tensor).any()--path:", node_feature_path)
            if torch.isnan(y_log_tensor).any():
                print("torch.isnan(y_log_tensor).any()--path:", node_feature_path)
            
            y_N_count = np.array(N_count_df.loc[[patch_name]])
            y_N_tensor = torch.tensor(y_N_count, dtype=torch.int)
            if torch.isinf(y_N_tensor).any():
                print("torch.isinf(y_N_tensor).any()--path:", node_feature_path)
            if torch.isnan(y_N_tensor).any():
                print("torch.isnan(y_N_tensor).any()--path:", node_feature_path)
            
            y_count_target = np.array(count_target_df.loc[[patch_name]])
            y_count_tensor = torch.tensor(y_count_target, dtype=torch.int)
            if torch.isinf(y_count_tensor).any():
                print("torch.isinf(y_count_tensor).any()--path:", node_feature_path)
            if torch.isnan(y_count_tensor).any():
                print("torch.isnan(y_count_tensor).any()--path:", node_feature_path)

            x_coor = np.array(spot_coord_tissue.loc[[patch_name]]['hires_row'])
            x_coor_tensor = torch.tensor(x_coor, dtype=torch.int)

            y_coor = np.array(spot_coord_tissue.loc[[patch_name]]['hires_col'])
            y_coor_tensor = torch.tensor(y_coor, dtype=torch.int)
            
            patch_img_path = os.path.join(patch_path, patch_name+'.png')
            patch_img_pil = Image.open(patch_img_path).convert("RGB")
            patch_img_tensor = patch_transform(patch_img_pil).unsqueeze(0)
            
            graph_data = Data(x=feature_tensor, edge_index=edge_index_tensor.t().contiguous(),\
                              edge_attr=edge_attr_tensor, pos=pos_tensor,\
                              y=y_log_tensor, N_count=y_N_tensor, count_target=y_count_tensor,\
                              x_coor=x_coor_tensor, y_coor=y_coor_tensor, patch_img=patch_img_tensor)
            
            graph_save_path = os.path.join(graph_img_root_path,tissue_name,patch_name)
            if not os.path.exists(graph_save_path):
                os.makedirs(graph_save_path)
            
            torch.save(graph_data, os.path.join(graph_save_path, 'graph_img_data.pt'))
            
            del graph_data