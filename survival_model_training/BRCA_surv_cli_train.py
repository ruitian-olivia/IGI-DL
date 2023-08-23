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

from surv_models.GAT_surv import BRCA_GAT_surv_cli as GAT_surv
from surv_clinical_training_function import train_deep_cox, test_cox_surv

class CoxGraphDataset(Dataset):
    # 可以把构建cox模型的其他一些clinical data指标也放进来
    def __init__(self, surv_df, transform=None, pre_transform=None):
        super(CoxGraphDataset, self).__init__()
        self.surv_df = surv_df

    def len(self):
        return len(self.surv_df)

    def get(self, idx):
        item_name = self.surv_df['HE_entity_submitter_id'].tolist()[idx]
        match_item = self.surv_df[self.surv_df["HE_entity_submitter_id"] == item_name]
        
        data_origin = torch.load(match_item['graph_path'].tolist()[0]) # filelist记录pt文件的位置
        transfer = T.ToSparseTensor()

        survival = match_item['Days'].tolist()[0]
        censor = match_item['Censor'].tolist()[0]
        age = match_item['Age'].tolist()[0]
        stage = match_item['Stage'].tolist()[0]
        item = match_item['HE_entity_submitter_id'].tolist()[0]

        data_re = Data(x=data_origin.x, edge_index=data_origin.edge_index)

        data = transfer(data_re)
        data.survival = torch.tensor(survival)
        data.censor = torch.tensor(censor)
        data.age = torch.tensor(age)
        data.stage = torch.tensor(stage)
        
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

def getGraphPath(HE_id, project_id, root_path):
    TCGA_project = project_id.split("-")[1]
    graph_path = os.path.join(root_path, TCGA_project, HE_id, '4.3_artifact_sophis_final.pt')
    if os.path.isfile(graph_path):
        return graph_path
    else:
        return np.nan     
    
def getDays(vital_status, days_to_death, days_to_last_follow_up):
    if vital_status == "Alive":
        days = days_to_last_follow_up
    elif vital_status == "Dead":
        days = days_to_death
    else:
        return np.nan
    
    if '--' not in days:
        return int(float(days))
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
    if gender == 'male':
        return 1
    elif gender == "female":
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
            
def Train(Argument):
    model_save_dir = os.path.join("model_weights", Argument.model_name)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        
    result_save_dir = os.path.join("model_result", Argument.model_name)
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
        
    surv_record_dir = os.path.join(result_save_dir,'surv_record')
    if not os.path.exists(surv_record_dir):
        os.makedirs(surv_record_dir)    
    
    BRCA_csv_path = '../super-patch_graph_construction/preprocessed_WSI/surv_csv/BRCA_surv.csv'
    surv_df = pd.read_csv(BRCA_csv_path)
      
    root_path = '../super-patch_graph_construction/preprocessed_WSI/supernode_graph'
    surv_df['graph_path'] = surv_df.apply(lambda x: getGraphPath(x.HE_entity_submitter_id, x.project_id, root_path), axis = 1)   
    surv_df['Days'] = surv_df.apply(lambda x: getDays(x.vital_status, x.days_to_death, x.days_to_last_follow_up), axis = 1)
    surv_df['Censor'] = surv_df.apply(lambda x: getCensor(x.vital_status), axis = 1)
    surv_df['Age'] = surv_df.apply(lambda x: getAge(x.age_at_index), axis = 1)
    surv_df['Stage'] = surv_df.apply(lambda x: getStage(x.ajcc_pathologic_stage), axis = 1)
    
    surv_df_filter = surv_df.dropna()
    surv_df_censor = surv_df_filter.loc[surv_df_filter['Censor'] == 1]
    print("All censored number:", len(surv_df_censor))
    surv_df_event = surv_df_filter.loc[surv_df_filter['Censor'] == 0]
    print("All death number", len(surv_df_event))
    
    for col_item in ['Censor', 'Stage']:
        print(col_item+":")
        print(surv_df_filter[col_item].value_counts())
        print(surv_df_filter[col_item].value_counts(normalize=True))
    print("The maean of age:", surv_df_filter['Age'].mean())
    print("The std of age:", surv_df_filter['Age'].std())
    
    # KM plot
    kmf = KaplanMeierFitter()
    kmf.fit(surv_df_filter['Days'], event_observed=surv_df_filter['Censor'])

    kmf.plot_survival_function(at_risk_counts=True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_save_dir,"CRC_KM_plot.png"))
    plt.clf()
    
    surv_df_censor = sklearn.utils.shuffle(surv_df_censor, random_state=Argument.random_seed) 
    surv_df_event = sklearn.utils.shuffle(surv_df_event, random_state=Argument.random_seed)
    
    runs = int(Argument.FF_number)  # Five-Fold validation
    print('runs:', runs)
    epochs = int(Argument.num_epochs)
    print('epochs:', epochs)
    
    train_loss = np.zeros((runs,epochs))
    train_acc = np.zeros((runs,epochs))
    
    age_beta = np.zeros((runs,epochs))
    stage_beta = np.zeros((runs,epochs))

    test_acc = np.zeros(runs)
    
    for Fi in range(runs):
        print("Fi:", Fi)
        trainFF_df = pd.concat([surv_df_censor[0:Fi * int(len(surv_df_censor) / 5)], surv_df_censor[(Fi + 1) * int(len(surv_df_censor) / 5):len(surv_df_censor)],\
            surv_df_event[0:Fi * int(len(surv_df_event) / 5)], surv_df_event[(Fi + 1) * int(len(surv_df_event) / 5):len(surv_df_event)]])
        validFF_df = pd.concat([surv_df_censor[Fi * int(len(surv_df_censor) / 5):(Fi + 1) * int(len(surv_df_censor) / 5)],\
            surv_df_event[Fi * int(len(surv_df_event) / 5):(Fi + 1) * int(len(surv_df_event) / 5)]])
            
        trainFF_df = sklearn.utils.shuffle(trainFF_df, random_state=1) 
        validFF_df = sklearn.utils.shuffle(validFF_df, random_state=1) 
        
        TrainDataset = CoxGraphDataset(surv_df=trainFF_df)
        ValidDataset = CoxGraphDataset(surv_df=validFF_df)
        print("len(TrainDataset):", TrainDataset)
        print("len(ValidDataset):", ValidDataset)
    
        # DataListLoader导入数据的方式需要在多GPU设备上运行
        train_loader = DataListLoader(TrainDataset, batch_size=len(TrainDataset), num_workers=1, pin_memory=True)
        val_loader = DataListLoader(ValidDataset, batch_size=len(ValidDataset), shuffle=True, num_workers=1, pin_memory=True,
                                drop_last=False)
    
        model = GAT_surv(Argument.dropout_rate, Argument.dropedge_rate, Argument) # GAT模型
        device = torch.device(int(Argument.gpu))
        model = DataParallel(model, device_ids=[0,1], output_device=0)  # 模型也是放在多设备上
        model = model.to(device)
    
        cox_loss = coxph_loss()
        cox_loss = cox_loss.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=Argument.learning_rate, weight_decay=Argument.weight_decay)
                
        for epoch in range(0, epochs):
            train_loss[Fi,epoch], train_acc[Fi,epoch] = train_deep_cox(model,train_loader,optimizer,
                                                    Argument,cox_loss)
            clinical_weight = torch.squeeze(model.module.risk_prediction_layer.weight)[:2]
            age_beta[Fi,epoch] = clinical_weight[0]
            stage_beta[Fi,epoch] = clinical_weight[1]      
            
            print("CV Run: {:03d}, Epoch: {:03d}, Train loss: {:.5f}, Train C-index: {:.4f}"\
              .format(Fi+1, epoch+1, train_loss[Fi,epoch], train_acc[Fi,epoch]))
            
            print("Age weight: {:.4f}, Stage weight: {:.4f}"\
              .format(clinical_weight[0], clinical_weight[1]))
            
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter in the model: %.2fM" % (total/1e6))
        
        torch.save(model.state_dict(), "{}/CV_{}.pth".format(model_save_dir,Fi))
        test_acc[Fi], test_surv_df = test_cox_surv(model, val_loader, cox_loss)
        print("CV Run Test: {:03d},  Test C-index: {:.5f}".format(Fi+1, test_acc[Fi]))
        
        test_surv_df.to_csv(os.path.join(surv_record_dir, 'run{}_test_surv.csv'.format(Fi+1)),
            sep=',',index=False,header=True)
        
    #求均值
    test_acc_mean = np.mean(test_acc)
    #求标准差
    test_acc_std = np.std(test_acc)
    print("===============================================")
    print("The mean of C-index in test set:%f" % test_acc_mean)
    print("The standard deviation of C-index in test set:%f" % test_acc_std)

    for run in range(runs):
        t_loss = train_loss[run][np.where(train_loss[run] > 0)]
        t_acc = train_acc[run][np.where(train_acc[run] > 0)]
        
        print("run %s print plot yes!" %run )
        fig = plt.figure(dpi=200,figsize=(15,8),facecolor='white')
        ax = fig.add_subplot(121)
        ax.set_title('Training  Loss (CV Fold {})'.format(run+1))
        ax.plot(range(1,len(t_loss)+1),t_loss, label='Training Loss')
        ax.set_xlabel('epochs')
        ax.set_ylabel('cox loss')
        ax.set_ylim(np.min(t_loss)-0.5,np.max(t_loss)+0.5) # consistent scale
        ax.legend()
        print("ax.legend()")

        bx = fig.add_subplot(122)
        bx.set_title('Training C-index (CV Fold {})'.format(run+1))
        bx.plot(range(1,len(t_acc)+1), t_acc)
        bx.set_xlabel('epochs')
        bx.set_ylabel('Value')
        bx.set_ylim(0, 1) # consistent scale
        bx.legend()
        print("bx.legend()")

        fig.savefig(os.path.join(result_save_dir, 'run{}_training_process.png'.format(run+1)), format='png')
        plt.close()

    training_record_dir = os.path.join(result_save_dir,'training_record')
    if not os.path.exists(training_record_dir):
        os.makedirs(training_record_dir)    

    np.savetxt(os.path.join(training_record_dir, 'train_loss_record.csv'), train_loss, delimiter=',', fmt='%.4f')
    np.savetxt(os.path.join(training_record_dir, 'train_acc_record.csv'), train_acc, delimiter=',', fmt='%.4f')
    np.savetxt(os.path.join(training_record_dir, 'age_beta_record.csv'), age_beta, delimiter=',', fmt='%.4f')
    np.savetxt(os.path.join(training_record_dir, 'stage_beta_record.csv'), stage_beta, delimiter=',', fmt='%.4f')
    np.savetxt(os.path.join(training_record_dir, 'test_acc_record.csv'), test_acc, delimiter=',', fmt='%.4f')

