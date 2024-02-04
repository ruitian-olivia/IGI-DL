# Training Integrated GIN+ViT model, with leave-one-patient validation to evaluate the performance
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

from GNN_CNN_fusion_model import GIN4layer_ViT
from GNN_CNN_training_function import setup_seed, train, valid, test, cal_gene_corr, test_spatial_visual
from pytorchtools import EarlyStopping

# model training arg parser
parser = argparse.ArgumentParser(description="Arguments for model training.")

parser.add_argument(
    "model_name",
    type=str,
    help="The name of the trained model",
)
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
    help="Dimension of the hidden layer in GNN model",
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
    "attn_heads",
    type=int,
    help="Attention heads number in ViT",
)
parser.add_argument(
    "dim_head",
    type=int,
    help="Dimension of head in ViT",
)
parser.add_argument(
    "hidden_features",
    type=int,
    help="Dimension of hidden features in ViT",
)
parser.add_argument(
    "out_features",
    type=int,
    help="Dimension of out features in ViT",
)
parser.add_argument(
    "--mlp_hidden",
    nargs='+',
    type=int,
    help="Dimension of MLP hidden layers",
)
args = parser.parse_args()

try:
    model_name = args.model_name
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    nhid = args.nhid
    epochs = args.epochs
    patience = args.patience
    attn_heads = args.attn_heads
    dim_head = args.dim_head
    hidden_features = args.hidden_features
    out_features = args.out_features
    mlp_hidden = args.mlp_hidden
except:
    print("error in parsing args")

setup_seed(100)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

if device == 'cpu':
    sys.exit()

tissue_list = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'sample6', 'sample7', 'sample8', 'sample9', 'sample10']
graph_pt_root_path = '../../preprocessed_data/filtered_graph_SVGs'
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

batch_size = 256
num_feature = 85
runs = len(tissue_list)

train_loss = np.zeros((runs,epochs))
val_loss = np.zeros((runs,epochs))
val_pear_corr = np.zeros((runs,epochs))
val_pear_logp = np.zeros((runs,epochs))
val_spea_corr = np.zeros((runs,epochs))
val_spea_logp = np.zeros((runs,epochs))
min_loss = 1e10*np.ones(runs)

test_loss = np.zeros(runs)
result_df_list = []

model_save_dir = os.path.join("model_weights", model_name)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

result_save_dir = os.path.join("model_result", model_name)
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)    

training_start = time.time()
epoch_counter = 0
for run in range(runs):
    dataload_start  = time.time()

    tissue_name = tissue_list[run]
    print("{} as test sample".format(tissue_name))
    test_graph_list = graph_dict[tissue_name]
    train_val_graph_list = []

    for key, value in graph_dict.items():
        if key != tissue_name:
            train_val_graph_list += value

    random.shuffle(train_val_graph_list)
    num_train_val = len(train_val_graph_list)
    num_train = int(num_train_val * 0.8)

    train_loader = DataLoader(train_val_graph_list[0:num_train], batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(train_val_graph_list[num_train:-1], batch_size=batch_size, shuffle = True)
    test_loader = DataLoader(test_graph_list, batch_size=batch_size, shuffle = False)

    dataload_end  = time.time()
    print("Run{:03d} data loader time: {:.2f}s ".format(run, dataload_end-dataload_start))

    model = GIN4layer_ViT(85, num_gene, nhid, attn_heads, dim_head, hidden_features, out_features, mlp_hidden).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    mse_loss = nn.MSELoss().to(device)     
    
    early_stopping = EarlyStopping(patience=patience, verbose=True, path="{}/model_{}_test_{}.pth".format(model_save_dir,model_name,tissue_name))

    for epoch in range(epochs):
        epoch_start  = time.time()
        loss = train(model,train_loader,optimizer,mse_loss,device) 
        epoch_train = time.time()
        train_loss[run,epoch] = loss
        val_loss[run,epoch], val_label, val_pred = valid(model,val_loader,mse_loss)
        epoch_val = time.time()
        
        val_result_df = cal_gene_corr(val_label, val_pred, predict_gene_list)
        val_pear_corr[run,epoch] = val_result_df["Pear_corr"].mean()
        val_pear_logp[run,epoch] = val_result_df["Pear_log_p"].mean()
        val_spea_corr[run,epoch] = val_result_df["Spea_corr"].mean()
        val_spea_logp[run,epoch] = val_result_df["Spea_log_p"].mean()

        epoch_counter += 1
        epoch_end  = time.time()
        print("Run: {:03d}, Epoch: {:03d}, Train time: {:.2f}s, Val time: {:.2f}s, Statistics time: {:.2f}s, Epoch time: {:.2f}s,\
              Train loss: {:.5f}, Val loss: {:.5f}, Val pearson corr: {:.5f}, Val pearson log_p:{:.5f}, Val spearman corr: {:.5f}, Val spearman log_p:{:.5f}"\
              .format(run+1, epoch+1, epoch_train-epoch_start, epoch_val-epoch_train, epoch_end-epoch_val, epoch_end-epoch_start,\
                    train_loss[run,epoch], val_loss[run,epoch],\
                    val_pear_corr[run,epoch], val_pear_logp[run,epoch],\
                    val_spea_corr[run,epoch], val_spea_logp[run,epoch]))
        
        early_stopping(val_loss[run, epoch], model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if val_loss[run,epoch] < min_loss[run]:
            min_loss[run] = val_loss[run,epoch]
            
    model.load_state_dict(torch.load("{}/model_{}_test_{}.pth".format(model_save_dir,model_name,tissue_name)))
    test_loss[run], test_label, test_pred, test_x_coor, test_y_coor = test(model,test_loader,mse_loss)
    print("Test tissue name: {},  Test MSE: {:.5f}".format(tissue_name, test_loss[run]))
    
    test_result_df = cal_gene_corr(test_label, test_pred, predict_gene_list)
    test_result_df.to_csv(os.path.join(result_save_dir,'Test_{}_result.csv'.format(tissue_name)), float_format='%.4f')
    result_df_list.append(test_result_df)

training_end = time.time()
training_time = training_end-training_start
print("Total training time: {:.2f}s \n Time/epoch: {:.2f}s".format(training_time, training_time/epoch_counter))

result_df_concat = pd.concat(result_df_list)
by_row_index = result_df_concat.groupby(result_df_concat.index)
result_df_means = by_row_index.mean()
result_df_std = by_row_index.std()
result_df_std.columns=['Pear_corr_std','Pear_log_p_std', 'Spea_corr_std','Spea_log_p_std']
result_df_all = pd.concat([result_df_means, result_df_std], axis=1)
pear_result_df = result_df_all.loc[:, ('Pear_corr', 'Pear_corr_std', 'Pear_log_p', 'Pear_log_p_std')]
pear_result_df.sort_values(by="Pear_corr", inplace=True, ascending=False)
pear_result_df.to_csv(os.path.join(result_save_dir,'result_df_Pearson.csv'), float_format='%.4f')
spea_result_df = result_df_all.loc[:, ('Spea_corr', 'Spea_corr_std', 'Spea_log_p', 'Spea_log_p_std')]
spea_result_df.sort_values(by="Spea_corr", inplace=True, ascending=False)
spea_result_df.to_csv(os.path.join(result_save_dir,'result_df_Spearman.csv'), float_format='%.4f')

result_mean = result_df_all.mean(axis=0)
result_median = result_df_all.median(axis=0)
print("------------Summary------------")
print("Result Mean:\n", result_mean)
print("Result Median:\n", result_median)

visual_gene_list = list(pear_result_df.index)[:2]

vis_start = time.time()
for tissue_name in tissue_list:
    model = GIN4layer_ViT(85, num_gene, nhid, attn_heads, dim_head, hidden_features, out_features, mlp_hidden).to(device)
    mse_loss = nn.MSELoss().to(device)
    model.load_state_dict(torch.load("{}/model_{}_test_{}.pth".format(model_save_dir,model_name,tissue_name)))
    test_graph_list = graph_dict[tissue_name]
    test_loader = DataLoader(test_graph_list, batch_size=batch_size, shuffle = False)
    
    _, test_label, test_pred, test_x_coor, test_y_coor = test(model,test_loader,mse_loss)
    
    for target_gene in visual_gene_list:
        target_pear_corr = pear_result_df.loc[target_gene, "Pear_corr"]
        fig_save_path = os.path.join(result_save_dir, "{:.3f}_{}".format(target_pear_corr,target_gene))
        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path)   
        
        gene_idx = predict_gene_list.index(target_gene)
        test_corr, _ = pearsonr(test_label[:, gene_idx], test_pred[:, gene_idx])
        
        test_result_dict = {
            "label":test_label[:, gene_idx],
            "pred":test_pred[:, gene_idx],
            "x_coor":test_x_coor,
            "y_coor":test_y_coor,
        } 
        test_result_df = pd.DataFrame(test_result_dict)
        test_spatial_visual(test_result_df, tissue_name, test_corr, target_gene, fig_save_path)
vis_end = time.time()
print("Gene expression visualization time: {:.2f}s".format(vis_end-vis_start))

for run in range(runs):
    tissue_name = tissue_list[run]

    t_loss = train_loss[run][np.where(train_loss[run] > 0)]
    v_loss = val_loss[run][np.where(val_loss[run] > 0)]

    pear_corr = val_pear_corr[run][np.where(val_pear_corr[run] > 0)]
    spea_corr = val_spea_corr[run][np.where(val_spea_corr[run] > 0)]
    
    fig = plt.figure(dpi=100,figsize=(15,8),facecolor='white')
    ax = fig.add_subplot(121)
    ax.set_title('Training & Validation Loss Trends (Except {})'.format(tissue_name))
    ax.plot(range(1,len(t_loss)+1),t_loss, label='Training Loss')
    ax.plot(range(1,len(v_loss)+1),v_loss,label='Validation Loss')
    minposs = np.where(v_loss == np.min(v_loss))[0][0] + 1
    ax.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_ylim(0, 2) # consistent scale
    ax.set_xlim(0, len(v_loss)+1) # consistent scale
    ax.legend()

    bx = fig.add_subplot(122)
    bx.set_title('Validation Corr (Run {})'.format(run))
    bx.plot(range(1,len(pear_corr)+1), pear_corr, label='Validation Pearson Corr')
    bx.plot(range(1,len(spea_corr)+1), spea_corr, label='Validation Spearman Corr')    
    bx.set_xlabel('epochs')
    bx.set_ylabel('Value')
    bx.set_ylim(0, 1) # consistent scale
    bx.legend()

    fig.savefig(os.path.join(result_save_dir, 'run{}_traning_process.png'.format(run)), format='png')
    plt.close()

training_record_dir = os.path.join(result_save_dir,'training_record')
if not os.path.exists(training_record_dir):
    os.makedirs(training_record_dir)    

np.savetxt(os.path.join(training_record_dir, 'train_loss_record.csv'), train_loss, delimiter=',', fmt='%.4f')
np.savetxt(os.path.join(training_record_dir, 'val_loss_record.csv'), val_loss, delimiter=',', fmt='%.4f')
np.savetxt(os.path.join(training_record_dir, 'val_pear_corr_record.csv'), val_pear_corr, delimiter=',', fmt='%.4f')
np.savetxt(os.path.join(training_record_dir, 'val_spea_corr_record.csv'), val_spea_corr, delimiter=',', fmt='%.4f')
np.savetxt(os.path.join(training_record_dir, 'test_loss_record.csv'), test_loss, delimiter=',', fmt='%.4f')