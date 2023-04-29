import os
import cv2
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from palettable.colorbrewer.diverging import RdYlBu_10_r

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(model,train_loader,optimizer,mse_loss,device):
    model.train()

    loss_all = 0
    for data in train_loader:
        img_data, y_label = data
        img_data = img_data.to(device)
        y_label = y_label.to(device)
        optimizer.zero_grad()
        output = model(img_data)
        loss = mse_loss(output, y_label)
        loss.backward()
        loss_all += len(img_data) * loss.item()
        optimizer.step()
    return loss_all / len(train_loader.dataset)

def valid(model,loader,mse_loss):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = 0.  
    
    label = np.array([])
    pred = np.array([])
    for data in loader:
        img_data, y_label = data
        img_data = img_data.to(device)
        y_label = y_label.to(device)
        output = model(img_data)
        
        loss_mean = mse_loss(output, y_label)
        loss += len(img_data) * loss_mean.item()
        
        _tmp_label = y_label.cpu().detach().numpy()
        _tmp_pred = output.cpu().detach().numpy()

        label = np.vstack([label,_tmp_label]) if label.size else _tmp_label
        pred = np.vstack([pred,_tmp_pred]) if pred.size else _tmp_pred

    return loss / len(loader.dataset), label, pred

def test(model,loader,mse_loss):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = 0.  
    
    label = np.array([])
    pred = np.array([])
    x_coor = np.array([])
    y_coor = np.array([])
    for data in loader:
        img_data, y_label, patch_name, x_coor_tensor, y_coor_tensor = data
        img_data = img_data.to(device)
        y_label = y_label.to(device)
        output = model(img_data)
        
        loss_mean = mse_loss(output, y_label)
        loss += len(img_data) * loss_mean.item()
        
        _tmp_label = y_label.cpu().detach().numpy()
        _tmp_pred = output.cpu().detach().numpy()
        _tmp_x_coor = x_coor_tensor.cpu().detach().numpy()
        _tmp_y_coor = y_coor_tensor.cpu().detach().numpy()

        label = np.vstack([label,_tmp_label]) if label.size else _tmp_label
        pred = np.vstack([pred,_tmp_pred]) if pred.size else _tmp_pred
        x_coor = np.vstack([x_coor,_tmp_x_coor]) if x_coor.size else _tmp_x_coor
        y_coor = np.vstack([y_coor,_tmp_y_coor]) if y_coor.size else _tmp_y_coor

    return loss / len(loader.dataset), label, pred, np.squeeze(x_coor), np.squeeze(y_coor)

def cal_gene_corr(label_df, pred_df, predict_gene_list):
    pear_corr_list = []
    pear_log_p_list = []
    spea_corr_list = []
    spea_log_p_list = []
    
    for idx in range(len(predict_gene_list)):
        label_gene = label_df[:,idx]
        pred_gene = pred_df[:,idx]
        
        pear_corr, pear_p = pearsonr(label_gene, pred_gene)
        pear_log_p = -np.log10(pear_p+1e-10)
        spea_corr, spea_p = spearmanr(label_gene, pred_gene)
        spea_log_p = -np.log10(spea_p+1e-10)
        
        pear_corr_list.append(pear_corr)
        pear_log_p_list.append(pear_log_p)
        spea_corr_list.append(spea_corr)
        spea_log_p_list.append(spea_log_p)
        
    result_dict = {"Pear_corr" : pear_corr_list,
                "Pear_log_p" : pear_log_p_list,
                "Spea_corr" : spea_corr_list,
                "Spea_log_p" : spea_log_p_list
                }
    result_df = pd.DataFrame(result_dict)
    result_df.index = predict_gene_list
    
    return result_df

def test_spatial_visual(test_result_df, tissue_name, test_corr, target_gene, save_path):
    l_minima = min(list(test_result_df["label"]))
    l_maxima = max(list(test_result_df["label"]))
    l_norm = matplotlib.colors.Normalize(vmin=l_minima, vmax=l_maxima, clip=True)
    l_mapper = cm.ScalarMappable(norm=l_norm, cmap=RdYlBu_10_r.mpl_colormap)
    
    pred_sr = test_result_df["pred"]
    q1 = pred_sr.quantile(0.25)
    q3 = pred_sr.quantile(0.75)
    iqr = q3-q1 
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    p_minima = max(fence_low, min(list(test_result_df["pred"])))
    p_maxima = min(fence_high,max(list(test_result_df["pred"])))
    p_norm = matplotlib.colors.Normalize(vmin=p_minima, vmax=p_maxima, clip=True)
    p_mapper = cm.ScalarMappable(norm=p_norm, cmap=RdYlBu_10_r.mpl_colormap)

    visium_root_dir = "../../dataset"
    visium_root_path = os.path.join(visium_root_dir, tissue_name)
    hires_img_path = os.path.join(visium_root_path, "spatial/tissue_hires_image.png")
    hires_img = cv2.imread(hires_img_path,1)
    hires_img_rgb = cv2.cvtColor(hires_img, cv2.COLOR_BGR2RGB)
    hires_img_shape = hires_img_rgb.shape

    scalefactor_file = os.path.join(visium_root_path, "spatial/scalefactors_json.json")
    with open(scalefactor_file, 'r', encoding = 'utf-8') as f:
        scalefactor_dict = json.load(f)

    fullres = scalefactor_dict['spot_diameter_fullres']
    scalef = scalefactor_dict['tissue_hires_scalef']
    spot_radius = round(fullres*scalef/2)

    l_heatmap_array = np.ones(hires_img_shape,dtype=np.uint8) * 255
    p_heatmap_array = np.ones(hires_img_shape,dtype=np.uint8) * 255
    mask_array = np.zeros(hires_img_shape[:2],dtype=np.uint8)

    for _,row in test_result_df.iterrows():
        x_coor = int(row['x_coor'])
        y_coor = int(row['y_coor'])
        l_feature = row['label']
        p_feature = row['pred']
        l_mapped_rgb = l_mapper.to_rgba(l_feature, alpha=None, bytes=True, norm=True)
        p_mapped_rgb = p_mapper.to_rgba(p_feature, alpha=None, bytes=True, norm=True)
        cv2.circle(l_heatmap_array,(y_coor,x_coor),spot_radius,(int(l_mapped_rgb[0]), int(l_mapped_rgb[1]), int(l_mapped_rgb[2])),-1)
        cv2.circle(p_heatmap_array,(y_coor,x_coor),spot_radius,(int(p_mapped_rgb[0]), int(p_mapped_rgb[1]), int(p_mapped_rgb[2])),-1)
        cv2.circle(mask_array,(y_coor,x_coor),spot_radius,255,-1)
        
    mask_inv = cv2.bitwise_not(mask_array)
    img_bg = cv2.bitwise_and(hires_img_rgb,hires_img_rgb,mask = mask_inv)
    l_img_fg = cv2.bitwise_and(l_heatmap_array,l_heatmap_array,mask = mask_array)
    p_img_fg = cv2.bitwise_and(p_heatmap_array,p_heatmap_array,mask = mask_array)
    l_out_img = cv2.add(img_bg,l_img_fg)
    p_out_img = cv2.add(img_bg,p_img_fg)

    fig = plt.figure(dpi=300,figsize=(15,8),facecolor='white')
    ax = fig.add_subplot(121)
    ax.set_title('True %s spatial expression (%s)' % (target_gene,tissue_name))
    l_img = plt.imshow(l_out_img)
    plt.axis('off')
    bx = fig.add_subplot(122)
    bx.set_title('Predicted %s (Correlation: %.3f)' % (target_gene,test_corr))
    p_img = plt.imshow(p_out_img, cmap=RdYlBu_10_r.mpl_colormap)
    plt.axis('off')
    cb1 = fig.colorbar(p_img, ax=[ax, bx], shrink=0.6)
    cb1.set_ticks([])
    fig.savefig(os.path.join(save_path, '{}_{}.png'.format(tissue_name, target_gene)), format='png')
    plt.close()
