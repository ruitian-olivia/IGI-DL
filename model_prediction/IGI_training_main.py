# Save the weights of IGI-DL (GIN+ResNet18) trained on the all training samples

import os
import cv2
import sys
import json
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
from torchvision import models
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader

from GNN_CNN_fusion_model import GIN4layer_ResNet18
from GNN_CNN_training_function import setup_seed, train, valid, cal_gene_corr
from pytorchtools import EarlyStopping

# model training arg parser
parser = argparse.ArgumentParser(description="Arguments for model training.")

parser.add_argument(
    "learning_rate",
    type=float,
    help="Learning rate",
)
parser.add_argument(
    "weight_decay", 
    type=float, 
    help="Weight decay"
)
parser.add_argument(
    "nhid",
    type=int,
    help="Dimension of the hidden layer",
)
parser.add_argument(
    "epochs",
    type=int,
    help="The number of epochs",
)
parser.add_argument(
    "patience",
    type=int,
    help="The number of patience",
)
parser.add_argument(
    "--mlp_hidden",
    nargs='+',
    type=int,
    help="Dimension of the MLP hidden layer",
)
args = parser.parse_args()

try:
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    nhid = args.nhid
    epochs = args.epochs
    patience = args.patience
    mlp_hidden = args.mlp_hidden
except:
    print("error in parsing args")

setup_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

if device=='cpu':
    sys.exit(0)

tissue_list = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'sample6']
graph_pt_root_path = '../../preprocessed_data/graph_SVGs'
graph_dict = {}

for tissue_name in tissue_list:
    tissue_graph_path = os.path.join(graph_pt_root_path,tissue_name)
    graph_tissue_list = []

    for patch_name in os.listdir(tissue_graph_path):
        graph_load_path = os.path.join(tissue_graph_path,patch_name,'graph_img_data.pt')
        graph_data = torch.load(graph_load_path)
        graph_tissue_list.append(graph_data)
    
    graph_dict.update({tissue_name: graph_tissue_list})

# Load the predicted gene names
predict_gene_path = '../../preprocessing/SVGs_SPARKX.txt'
with open(predict_gene_path, "r", encoding="utf-8") as f:
    predict_gene_list = f.read().splitlines()
num_gene = len(predict_gene_list)

batch_size = 512
num_feature = 85

train_loss = np.zeros((epochs))
val_loss = np.zeros((epochs))
val_pear_corr = np.zeros((epochs))
val_pear_logp = np.zeros((epochs))
val_spea_corr = np.zeros((epochs))
val_spea_logp = np.zeros((epochs))
min_loss = 1e10

train_val_graph_list = []

for key, value in graph_dict.items():
    if key != tissue_name:
        train_val_graph_list += value

random.shuffle(train_val_graph_list)
num_train_val = len(train_val_graph_list)
num_train = int(num_train_val * 0.8)

train_loader = DataLoader(train_val_graph_list[0:num_train], batch_size=batch_size, shuffle = True)
val_loader = DataLoader(train_val_graph_list[num_train:-1], batch_size=batch_size, shuffle = True)

model = GIN4layer_ResNet18(85, num_gene, nhid, mlp_hidden).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
mse_loss = nn.MSELoss().to(device)     
    
early_stopping = EarlyStopping(patience=patience, verbose=True, path="IGI-DL-weights.pth")

for epoch in range(epochs):
    loss = train(model,train_loader,optimizer,mse_loss,device) 
    epoch_train = time.time()
    train_loss[epoch] = loss
    val_loss[epoch], val_label, val_pred = valid(model,val_loader,mse_loss)
    
    val_result_df = cal_gene_corr(val_label, val_pred, predict_gene_list)
    val_pear_corr[epoch] = val_result_df["Pear_corr"].mean()
    val_pear_logp[epoch] = val_result_df["Pear_log_p"].mean()
    val_spea_corr[epoch] = val_result_df["Spea_corr"].mean()
    val_spea_logp[epoch] = val_result_df["Spea_log_p"].mean()

    print("Epoch: {:03d}, Train loss: {:.5f}, Val loss: {:.5f}, Val pearson corr: {:.5f}, Val pearson log_p:{:.5f}, Val spearman corr: {:.5f}, Val spearman log_p:{:.5f}"\
            .format(epoch+1, \
                train_loss[epoch], val_loss[epoch],\
                val_pear_corr[epoch], val_pear_logp[epoch],\
                val_spea_corr[epoch], val_spea_logp[epoch]))
        
    early_stopping(val_loss[epoch], model)
    if early_stopping.early_stop:
        print("Early stopping")
        break
    if val_loss[epoch] < min_loss:
        min_loss = val_loss[epoch]
 