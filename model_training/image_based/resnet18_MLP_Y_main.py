# Training Image-based ResNet18 model, with leave-one-patient validation to evaluate the performance
import os
import cv2
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
from scipy.stats import pearsonr
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from torchvision import transforms, models
from torch.utils.data import random_split, DataLoader

from patch_Data_Loader import patch_train_data, patch_test_data
from CNN_model import ResNet_MLP_gene
from CNN_training_function import setup_seed, train, valid, test, cal_gene_pearson, mape, test_spatial_visual
from pytorchtools import EarlyStopping

# model training arg parser
parser = argparse.ArgumentParser(description="Arguments for model training.")

parser.add_argument(
    "model_name",
    type=str,
    help="The name of the trainned model",
)
parser.add_argument(
    "imagenet_flag",
    type=bool
)
parser.add_argument(
    "learning_rate",
    type=float
)
parser.add_argument(
    "weight_decay", 
    type=float,
)
parser.add_argument(
    "corr_thresh",
    type=float,
    help="Threshold of gene correlation",
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
    model_name = args.model_name
    imagenet_flag = args.imagenet_flag
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    corr_thresh = args.corr_thresh
    epochs = args.epochs
    patience = args.patience
    mlp_hidden = args.mlp_hidden
except:
    print("error in parsing args")

setup_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

tissue_list = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5']
patch_root_path = "../../preprocessed_data/HE_nmzd"

# Load the predicted gene names
predict_gene_path = '../../preprocessing/predict_gene_list.txt'
with open(predict_gene_path, "r", encoding="utf-8") as f:
    predict_gene_list = f.read().splitlines()
num_gene = len(predict_gene_list)

batch_size = 512
runs = len(tissue_list)

train_loss = np.zeros((runs,epochs))
val_loss = np.zeros((runs,epochs))
val_corr = np.zeros((runs,epochs))
val_log_p = np.zeros((runs,epochs))
val_mape = np.zeros((runs,epochs))
min_loss = 1e10*np.ones(runs)

test_loss = np.zeros(runs)
result_df_list = []

model_save_dir = os.path.join("model_weights", model_name)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

result_save_dir = os.path.join("model_result", model_name)
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)

patch_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) 
])

training_start = time.time()
epoch_counter = 0
for run in range(runs):
    dataload_start  = time.time()

    tissue_name = tissue_list[run]
    print("{} as test sample".format(tissue_name))
    train_patch_dataset = patch_train_data(root=patch_root_path, exclude_tissue= tissue_name, transform=patch_transform)
    train_size = int(0.8 * len(train_patch_dataset))
    val_size = len(train_patch_dataset) - train_size
    train_dataset, val_dataset = random_split(train_patch_dataset, [train_size, val_size])
    print("The number of all training patches:", len(train_dataset))
    print("The number of all validation patches:", len(val_dataset))

    test_dataset = patch_test_data(root=patch_root_path, test_tissue= tissue_name, transform=patch_transform)
    print("The number of all test patches:", len(test_dataset))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    dataload_end  = time.time()
    print("Run {:03d} data loader time: {:.2f}s ".format(run+1, dataload_end-dataload_start))

    model = ResNet_MLP_gene(gene_dim=num_gene, mlp_hidden_list=mlp_hidden, model_type='resnet18').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

    if imagenet_flag:
        print("Yes:", imagenet_flag)
        resnet18 = models.resnet18(pretrained=True).to(device)
        resnet18.eval()
        pretrained_dict = resnet18.state_dict()

        model_dict = model.state_dict()
        model_keys = []
        for k, v in model_dict.items():
            model_keys.append(k)

        model_resnet_dict = model.resnet.state_dict()
        model_resnet_keys = []
        for k, v in model_resnet_dict.items():
            model_resnet_keys.append(k)
        
        i = 0
        for k, v in pretrained_dict.items():
            model_dict[model_keys[i]] = v
            i += 1
            if i >= len(model_resnet_keys):
                break

        model.load_state_dict(model_dict)
    else:
        print("No:", imagenet_flag)

    mse_loss = nn.MSELoss().to(device)     
    
    early_stopping = EarlyStopping(patience=patience, verbose=True, path="{}/model_{}_test_{}.pth".format(model_save_dir,model_name,tissue_name))

    for epoch in range(epochs):
        epoch_start  = time.time()

        loss = train(model,train_loader,optimizer,mse_loss,device) 
        train_loss[run,epoch] = loss
        val_loss[run,epoch], val_label, val_pred = valid(model,val_loader,mse_loss)
        
        val_result_df = cal_gene_pearson(val_label, val_pred, predict_gene_list)
        val_corr[run,epoch] = val_result_df["Correlation"].mean()
        val_log_p[run,epoch] = val_result_df["Log_p_value"].mean()
        val_mape[run,epoch] = val_result_df["MAPE"].mean()

        epoch_counter += 1
        epoch_end  = time.time()
        print("Run: {:03d}, Epoch: {:03d}, Train time: {:.2f}s, Train loss: {:.5f}, Val loss: {:.5f}, Val corr: {:.5f}, Val log_p:{:.5f}, Val MAPE:{:.5f}"\
              .format(run+1, epoch+1, epoch_end-epoch_start, train_loss[run,epoch], val_loss[run,epoch],\
                      val_corr[run,epoch], val_log_p[run,epoch], val_mape[run,epoch]))
        
        early_stopping(val_loss[run, epoch], model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if val_loss[run,epoch] < min_loss[run]:
            min_loss[run] = val_loss[run,epoch]
    
    model.load_state_dict(torch.load("{}/model_{}_test_{}.pth".format(model_save_dir,model_name,tissue_name)))
    test_loss[run], test_label, test_pred, test_x_coor, test_y_coor = test(model,test_loader,mse_loss)
    print("Test tissue name: {},  Test MSE: {:.5f}".format(tissue_name, test_loss[run]))
    
    test_result_df = cal_gene_pearson(test_label, test_pred, predict_gene_list)
    test_result_df.to_csv(os.path.join(result_save_dir,'Test_{}_result.csv'.format(tissue_name)), float_format='%.4f')
    result_df_list.append(test_result_df)

training_end = time.time()
training_time = training_end-training_start
print("Total training time: {:.2f}s \n Time/epoch: {:.2f}s".format(training_time, training_time/epoch_counter))

result_df_concat = pd.concat(result_df_list)
by_row_index = result_df_concat.groupby(result_df_concat.index)
result_df_means = by_row_index.mean()
result_df_std = by_row_index.std()
result_df_std.columns=['Corr_std','Log_p_std','MAPE_std']
result_df_all = pd.concat([result_df_means, result_df_std], axis=1)
column_order = ['Correlation', 'Corr_std', 'Log_p_value', 'Log_p_std', 'MAPE', 'MAPE_std']
result_df_all = result_df_all[column_order]
result_df_all.sort_values(by="Correlation", inplace=True, ascending=False)
result_df_all.to_csv(os.path.join(result_save_dir,'result_df_all.csv'), float_format='%.4f')

result_mean = result_df_all.mean(axis=0)
print("------------Summary------------")
print("Result Mean:\n", result_mean)

result_visual_df = result_df_all[result_df_all['Correlation'] > corr_thresh] 
visual_gene_list = list(result_visual_df.index)
print("The ratio of gene correlation greater than {:.3f}: {:.3f}".format(corr_thresh, len(visual_gene_list)/num_gene))

vis_start = time.time()
for tissue_name in tissue_list:
    model = ResNet_MLP_gene(gene_dim=num_gene, mlp_hidden_list=mlp_hidden, model_type='resnet18').to(device)
    mse_loss = nn.MSELoss().to(device)
    model.load_state_dict(torch.load("{}/model_{}_test_{}.pth".format(model_save_dir,model_name,tissue_name)))
    
    test_dataset = patch_test_data(root=patch_root_path, test_tissue=tissue_name, transform=patch_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    _, test_label, test_pred, test_x_coor, test_y_coor = test(model,test_loader,mse_loss)
    
    for target_gene in visual_gene_list:
        target_gene_corr = result_visual_df.loc[target_gene, "Correlation"]
        fig_save_path = os.path.join(result_save_dir, "{:.3f}_{}".format(target_gene_corr,target_gene))
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

    corr = val_corr[run][np.where(val_corr[run] > 0)]
    mape = val_mape[run][np.where(val_mape[run] > 0)]

    fig = plt.figure(dpi=300,figsize=(15,8),facecolor='white')
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
    bx.set_title('Validation Correlation & MAPE (Run {})'.format(run))
    bx.plot(range(1,len(corr)+1),corr, label='Validation Correlation')
    bx.plot(range(1,len(mape)+1),mape,label='Validation MAPE')
    bx.set_xlabel('epochs')
    bx.set_ylabel('Value')
    bx.set_ylim(0, 1) # consistent scale
    bx.set_xlim(0, len(mape)+1) # consistent scale
    bx.legend()

    fig.savefig(os.path.join(result_save_dir, 'run{}_traning_process.png'.format(run)), format='png')
    plt.close()

training_record_dir = os.path.join(result_save_dir,'training_record')
if not os.path.exists(training_record_dir):
    os.makedirs(training_record_dir)    

np.savetxt(os.path.join(training_record_dir, 'train_loss_record.csv'), train_loss, delimiter=',', fmt='%.4f')
np.savetxt(os.path.join(training_record_dir, 'val_loss_record.csv'), val_loss, delimiter=',', fmt='%.4f')
np.savetxt(os.path.join(training_record_dir, 'val_corr_record.csv'), val_corr, delimiter=',', fmt='%.4f')
np.savetxt(os.path.join(training_record_dir, 'val_mape_record.csv'), val_mape, delimiter=',', fmt='%.4f')
np.savetxt(os.path.join(training_record_dir, 'test_loss_record.csv'), test_loss, delimiter=',', fmt='%.4f')
 