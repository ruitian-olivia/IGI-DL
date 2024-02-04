import os
import copy
import pytz
import torch
import sklearn
import datetime
import argparse
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.nn import DataParallel
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.transforms import Polar
from torch_geometric.loader import DataListLoader, DataLoader
from torch.utils.data.sampler import Sampler
from torch_geometric.nn import GATConv as GATConv_v1
from torch_geometric.nn import GATv2Conv as GATConv
import torch.nn.functional as F
import torch_geometric.transforms as T
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from surv_models.GAT_surv import CRC_GAT_surv_cli as GAT_surv
from surv_clinical_training_function import test_LOOV_cox_surv

class CoxGraphDataset(Dataset):
    # 可以把构建cox模型的其他一些clinical data指标也放进来
    def __init__(self, surv_df, transform=None, pre_transform=None):
        super(CoxGraphDataset, self).__init__()
        self.surv_df = surv_df

    def len(self):
        return len(self.surv_df)

    def get(self, idx):
        item_name = self.surv_df['ParentSpecimen'].tolist()[idx]
        match_item = self.surv_df[self.surv_df["ParentSpecimen"] == item_name]
        
        data_origin = torch.load(match_item['graph_path'].tolist()[0]) # filelist记录pt文件的位置
        transfer = T.ToSparseTensor()

        survival = match_item['SurvivalTime'].tolist()[0]
        censor = match_item['Censor'].tolist()[0]
        age = match_item['Age'].tolist()[0]
        gender = match_item['Gender'].tolist()[0]
        stage = match_item['Stage'].tolist()[0]
        cancer_type = match_item['Type'].tolist()[0]
        item = match_item['ParentSpecimen'].tolist()[0]

        data_re = Data(x=data_origin.x, edge_index=data_origin.edge_index)

        data = transfer(data_re)
        data.survival = torch.tensor(survival)
        data.censor = torch.tensor(censor)
        data.age = torch.tensor(age)
        data.gender = torch.tensor(gender)
        data.stage = torch.tensor(stage)
        data.cancer_type = torch.tensor(cancer_type)
        
        data.item = item
        data.edge_attr = data_origin.edge_attr
        data.pos = data_origin.pos

        return data

class coxph_loss(torch.nn.Module):

    def __init__(self):
        super(coxph_loss, self).__init__()

    def forward(self, risk, censors):
        
        riskmax = F.normalize(risk, p=2, dim=0)

        log_risk = torch.log((torch.cumsum(torch.exp(riskmax), dim=0)))

        uncensored_likelihood = torch.add(riskmax, -log_risk)
        resize_censors = censors.resize_(uncensored_likelihood.size()[0], 1)
        censored_likelihood = torch.mul(uncensored_likelihood, resize_censors)

        loss = -torch.sum(censored_likelihood) / float(censors.nonzero().size(0))

        return loss

def getGraphPath(HE_id, root_path):
    graph_path = os.path.join(root_path, HE_id, '4.3_artifact_sophis_final.pt')
    if os.path.isfile(graph_path):
        return graph_path
    else:
        return np.nan     
    
def getCensor(vital_status):
    if vital_status == "Alive":
        return 0
    elif vital_status == "Dead":
        return 1
    else:
        return np.nan

def getAge(age_at_index):
    if isinstance(age_at_index, int):
        return age_at_index
    else:
        return np.nan
    
def getGender(gender):
    if gender == 'Male':
        return 1
    elif gender == "Female":
        return 0
    else:
        return np.nan
    
def getStage(ajcc_pathologic_stage):
    if ('X' in ajcc_pathologic_stage) or ('IV' in ajcc_pathologic_stage) or ('III' in ajcc_pathologic_stage):
        return 1
    elif ("II" in ajcc_pathologic_stage) or ("I" in ajcc_pathologic_stage):
        return 0
    else:
        return np.nan
        
def getType(project_id):
    if 'READ' in project_id:
        return 0
    else:
        return 1
    
def External_test(Argument):
        
    result_save_dir = os.path.join("model_result", Argument.model_name)
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
        
    surv_record_dir = os.path.join(result_save_dir,'surv_record')
    if not os.path.exists(surv_record_dir):
        os.makedirs(surv_record_dir)    
    
    MCO_csv_path = '../super-patch_graph_construction/preprocessed_MCO/surv_csv/MCO_surv.csv'
    surv_df = pd.read_csv(MCO_csv_path)
    print("surv_df.shape:", surv_df.shape)
      
    root_path = '../super-patch_graph_construction/preprocessed_MCO/supernode_graph'
    surv_df['graph_path'] = surv_df.apply(lambda x: getGraphPath(x.ParentSpecimen, root_path), axis = 1)   
    surv_df['Censor'] = surv_df.apply(lambda x: getCensor(x.VitalStatus), axis = 1)
    surv_df['Age'] = surv_df.apply(lambda x: getAge(x.Age), axis = 1)
    surv_df['Gender'] = surv_df.apply(lambda x: getGender(x.Sex), axis = 1)
    surv_df['Stage'] = surv_df.apply(lambda x: getStage(x.OverallStage), axis = 1)
    surv_df['Type'] = surv_df.apply(lambda x: getType(x.Type), axis = 1)
    
    surv_df_filter = surv_df.dropna()
    print("surv_df_filter.shape:", surv_df_filter.shape)
    
    for col_item in ['Censor', 'Gender', 'Stage', 'Type']:
        print(col_item+":")
        print(surv_df_filter[col_item].value_counts())
        print(surv_df_filter[col_item].value_counts(normalize=True))
    print("The maean of age:", surv_df_filter['Age'].mean())
    print("The std of age:", surv_df_filter['Age'].std())
    
    # KM plot
    kmf = KaplanMeierFitter()
    kmf.fit(surv_df_filter['SurvivalTime'], event_observed=surv_df_filter['Censor'])

    kmf.plot_survival_function(at_risk_counts=True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_save_dir,"MCO_KM_plot.png"))
    plt.clf()
    
    TestDataset = CoxGraphDataset(surv_df=surv_df_filter)
    print("len(TestDataset):", TestDataset)

    test_loader = DataListLoader(TestDataset, batch_size=64, shuffle=False, num_workers=1, pin_memory=True,
                            drop_last=False)

    model = GAT_surv(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    model = DataParallel(model, device_ids=[0,1], output_device=0)
    
    weights_root = './model_weights'
    weights_path = os.path.join(weights_root, Argument.model_weights_name, 'ALL_model.pth')

    model.load_state_dict(torch.load(weights_path))
    
    device = torch.device(int(Argument.gpu))
    
    model = model.to(device)

    cox_loss = coxph_loss()
    cox_loss = cox_loss.to(device)

    test_acc, test_surv_df = test_LOOV_cox_surv(model, test_loader, cox_loss)
    print("MCO Test C-index: {:.5f}".format(test_acc))
    
    test_surv_df.to_csv(os.path.join(surv_record_dir, 'MCO_test_surv.csv'),
        sep=',',index=False,header=True)
        