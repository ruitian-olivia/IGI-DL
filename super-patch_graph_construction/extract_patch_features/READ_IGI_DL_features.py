import os
import cv2
import sys
import json
import time
import math
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
import multiprocessing
from torchvision import models
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader

from GNN_CNN_fusion_model import GIN4layer_ResNet18_visual

def predict_gene_features(model, patient_loader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fusion_latent_record = None
    gene_predict_record = None

    for data in patient_loader:
        data = data.to(device)
        fusion_output, gene_output = model(data)
                
        _tmp_fusion = fusion_output.cpu().detach().numpy()
        _tmp_gene = gene_output.cpu().detach().numpy()
        _tmp_x_coord = data.x_coor.cpu().detach().numpy()
        _tmp_y_coord = data.y_coor.cpu().detach().numpy()
        
        if fusion_latent_record is None:
            fusion_latent_record = _tmp_fusion
            gene_predict_record = _tmp_gene
            x_coord_record = _tmp_x_coord
            y_coord_record = _tmp_y_coord

        else:
            fusion_latent_record = np.vstack([fusion_latent_record, _tmp_fusion])
            gene_predict_record = np.vstack([gene_predict_record, _tmp_gene])
            x_coord_record = np.hstack([x_coord_record, _tmp_x_coord])
            y_coord_record = np.hstack([y_coord_record, _tmp_y_coord])
    
    fusion_latent_df = pd.DataFrame(fusion_latent_record)
    gene_latent_df = pd.DataFrame(gene_predict_record)
    x_coord_df = pd.DataFrame(x_coord_record)
    y_coord_df = pd.DataFrame(y_coord_record)
    
    return fusion_latent_df, gene_latent_df, x_coord_df, y_coord_df

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

# Load the predicted gene names
predict_gene_path = '../../model_weights/CRC/CRC_gene_list.txt'
with open(predict_gene_path, "r", encoding="utf-8") as f:
    predict_gene_list = f.read().splitlines()
num_gene = len(predict_gene_list)

model_weight_path = '../../model_weights/CRC/IGI-DL-CRC-weights.pth'
model = GIN4layer_ResNet18_visual(85, num_gene, 256, [512, 256, 256]).to(device)
model.load_state_dict(torch.load(model_weight_path, map_location=torch.device(device)))

model_name = 'IGI_DL_READ'

graph_pt_root_path = '../preprocessed_WSI/graph_image/READ'
patient_list = os.listdir(graph_pt_root_path)
print("len(patient_list):", len(patient_list))

for patient_name in patient_list:
    patient_path = os.path.join(graph_pt_root_path, patient_name)
    graph_patch_list = []
    barcodes_list = []

    for tile_name in os.listdir(patient_path):
        graph_load_path = os.path.join(patient_path, tile_name, 'graph_img_data.pt')
        graph_img_data = torch.load(graph_load_path)
        graph_patch_list.append(graph_img_data)
        barcodes_list.append(tile_name)

    graph_loader = DataLoader(graph_patch_list, batch_size = 128, shuffle = False)
    fusion_latent, gene_latent, x_coord_df, y_coord_df = predict_gene_features(model, graph_loader)
    
    fusion_latent['barcodes'] = barcodes_list
    gene_latent['barcodes'] = barcodes_list
    
    fusion_latent['x_coord'] = x_coord_df
    gene_latent['x_coord'] = x_coord_df
    
    fusion_latent['y_coord'] = y_coord_df
    gene_latent['y_coord'] = y_coord_df
        
    output_path = os.path.join(model_name, patient_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    fusion_latent.to_csv(os.path.join(output_path, "fusion_latent.csv"), index=False, float_format='%.3f')
    gene_latent.to_csv(os.path.join(output_path, "gene_latent.csv"), index=False, float_format='%.3f')