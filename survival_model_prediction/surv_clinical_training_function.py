import os
import cv2
import json
import glob
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import lifelines.utils.concordance as LUC
from palettable.colorbrewer.diverging import RdYlBu_10_r

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def cox_sort(out, tempsurvival, tempcensor, tempID,
             EpochRisk, EpochSurv, EpochCensor, EpochID):

    # out是什么样的数据格式呢？ pytorch tensor的格式
    sort_idx = torch.argsort(tempsurvival, descending=True) # 返回排序后的值所对应的下标，即torch.sort()返回的indices (递减，降序)
    updated_feature_list = []

    risklist = out[sort_idx]
    tempsurvival = tempsurvival[sort_idx]
    tempcensor = tempcensor[sort_idx] 
    for idx in sort_idx.cpu().detach().tolist():
        EpochID.append(tempID[idx])

    risklist = risklist.to(out.device)
    tempsurvival = tempsurvival.to(out.device)
    tempcensor = tempcensor.to(out.device)

    for riskval, survivalval, censorval in zip(risklist,
                                    tempsurvival, tempcensor):
        EpochRisk.append(riskval.cpu().detach().item())
        EpochSurv.append(survivalval.cpu().detach().item())
        EpochCensor.append(censorval.cpu().detach().item())

    return (risklist, tempsurvival, tempcensor),\
        (EpochRisk, EpochSurv, EpochCensor)
        
def cox_LOOV_sort(out, tempsurvival, tempcensor, tempID,
             EpochRisk, EpochSurv, EpochCensor, EpochID):

    # out是什么样的数据格式呢？ pytorch tensor的格式
    sort_idx = torch.argsort(tempsurvival, descending=True) # 返回排序后的值所对应的下标，即torch.sort()返回的indices (递减，降序)
    updated_feature_list = []

    risklist = out[sort_idx]
    tempsurvival = tempsurvival[sort_idx]
    tempcensor = tempcensor[sort_idx]
    for idx in sort_idx.cpu().detach().tolist():
        EpochID.append(tempID[idx])

    risklist = risklist.to(out.device)
    tempsurvival = tempsurvival.to(out.device)
    tempcensor = tempcensor.to(out.device)

    for riskval, survivalval, censorval, IDval in zip(risklist,
                                    tempsurvival, tempcensor, tempID):
        EpochRisk.append(riskval.cpu().detach().item())
        EpochSurv.append(survivalval.cpu().detach().item())
        EpochCensor.append(censorval.cpu().detach().item())

    return (risklist, tempsurvival, tempcensor),\
        (EpochRisk, EpochSurv, EpochCensor, EpochID)
        
def accuracytest(survivals, risk, censors): # 计算生存分析模型的准确度
    survlist = []
    risklist = []
    censorlist = []

    for riskval in risk:
        risklist.append(riskval.cpu().detach().item())

    for censorval in censors:
        censorlist.append(censorval.cpu().detach().item())

    for surval in survivals:
        survlist.append(surval.cpu().detach().item())

    C_value = LUC.concordance_index(survlist, -np.exp(risklist), censorlist) # 评价指标 C-index

    return C_value

def train_deep_cox(model,train_loader,optimizer,Argument,cox_loss):
    model.train()
    grad_flag = True
    
    with torch.set_grad_enabled(grad_flag):
        loss = 0
        EpochRisk = []
        EpochSurv = []
        EpochCensor = []
        EpochID = []
        Epochloss = 0
        batchcounter = 0
        
        for c, d in enumerate(train_loader, 1):
            optimizer.zero_grad()
            
            out = model(d)
            tempsurvival = torch.tensor([data.survival for data in d]) # 单个batch数据中的survival data
            tempcensor = torch.tensor([data.censor for data in d]) # 0 删失 1 死亡
            tempID = np.asarray([data.item for data in d]) # item='TCGA-F5-6464-01Z-00-DX1'
            
            tempSet, EpochSet = \
                cox_sort(out, tempsurvival, tempcensor, tempID,
                    EpochRisk, EpochSurv, EpochCensor, EpochID)
                
            risklist, tempsurvival, tempcensor = tempSet
            EpochRisk, EpochSurv, EpochCensor = EpochSet
            
            loss = cox_loss(risklist, tempcensor)
                   
            loss.backward()
            optimizer.step()
                        
            Epochloss += loss.cpu().detach().item()
            batchcounter += 1
            
        Epochloss = Epochloss / batchcounter
        Epochacc = accuracytest(torch.tensor(EpochSurv), torch.tensor(EpochRisk), torch.tensor(EpochCensor))
        
        return Epochloss, Epochacc

def train_deep_cox_step(model,train_loader,optimizer,Argument,cox_loss):
    model.train()
    grad_flag = True
    
    with torch.set_grad_enabled(grad_flag):
        loss = 0
        EpochRisk = []
        EpochSurv = []
        EpochCensor = []
        EpochID = []
        Epochloss = 0
        batchcounter = 0
        
        for c, d in enumerate(train_loader, 1):
            optimizer.zero_grad()
            
            out = model(d)
            tempsurvival = torch.tensor([data.survival for data in d]) # 单个batch数据中的survival data
            tempcensor = torch.tensor([data.censor for data in d]) # 0 删失 1 死亡
            tempID = np.asarray([data.item for data in d]) # item='TCGA-F5-6464-01Z-00-DX1'
            
            tempSet, EpochSet = \
                cox_sort(out, tempsurvival, tempcensor, tempID,
                    EpochRisk, EpochSurv, EpochCensor, EpochID)
                
            risklist, tempsurvival, tempcensor = tempSet
            EpochRisk, EpochSurv, EpochCensor = EpochSet
            
            loss = cox_loss(risklist, tempcensor)
                   
            loss.backward()
            optimizer.step()
            
            Epochloss += loss.cpu().detach().item()
            batchcounter += 1
            
        Epochloss = Epochloss / batchcounter
        Epochacc = accuracytest(torch.tensor(EpochSurv), torch.tensor(EpochRisk), torch.tensor(EpochCensor))
        
        return Epochloss, Epochacc, batchcounter
  
def test_deep_cox(model,test_loader,cox_loss):
    model.eval()
    grad_flag = False
    
    with torch.set_grad_enabled(grad_flag):
        loss = 0
        EpochRisk = []
        EpochSurv = []
        EpochCensor = []
        EpochID = []
        Epochloss = 0
        
        for c, d in enumerate(test_loader, 1):       
            out = model(d)
            tempsurvival = torch.tensor([data.survival for data in d]) # 单个batch数据中的survival data
            tempcensor = torch.tensor([data.censor for data in d]) # 0 删失 1 死亡
            tempID = np.asarray([data.item for data in d]) # item='TCGA-F5-6464-01Z-00-DX1'

            tempSet, EpochSet = \
                cox_sort(out, tempsurvival, tempcensor, tempID,
                    EpochRisk, EpochSurv, EpochCensor, EpochID)

            risklist, tempsurvival, tempcensor = tempSet
            EpochRisk, EpochSurv, EpochCensor = EpochSet
              
        Epochacc = accuracytest(torch.tensor(EpochSurv), torch.tensor(EpochRisk), torch.tensor(EpochCensor))
   
        return Epochacc
    
def test_cox_surv(model,test_loader,cox_loss):
    model.eval()
    grad_flag = False
    
    with torch.set_grad_enabled(grad_flag):
        loss = 0
        EpochRisk = []
        EpochSurv = []
        EpochCensor = []
        EpochID = []
        Epochloss = 0
        
        for c, d in enumerate(test_loader, 1):
            out = model(d)
            tempsurvival = torch.tensor([data.survival for data in d]) # 单个batch数据中的survival data
            tempcensor = torch.tensor([data.censor for data in d]) # 0 删失 1 死亡
            tempID = np.asarray([data.item for data in d]) # item='TCGA-F5-6464-01Z-00-DX1'

            tempSet, EpochSet = \
                cox_sort(out, tempsurvival, tempcensor, tempID,
                    EpochRisk, EpochSurv, EpochCensor, EpochID)

            risklist, tempsurvival, tempcensor = tempSet
            EpochRisk, EpochSurv, EpochCensor = EpochSet
              
        Epochacc = accuracytest(torch.tensor(EpochSurv), torch.tensor(EpochRisk), torch.tensor(EpochCensor))
        surv_df = pd.DataFrame(list(zip(EpochRisk, EpochSurv, EpochCensor)),
                               columns = ['riskScore', 'Surv', 'Censor'])
           
        return Epochacc, surv_df
    
def test_LOOV_cox_surv(model,test_loader,cox_loss):
    model.eval()
    grad_flag = False
    
    with torch.set_grad_enabled(grad_flag):
        loss = 0
        EpochRisk = []
        EpochSurv = []
        EpochCensor = []
        EpochID = []
        Epochloss = 0
        
        for c, d in enumerate(test_loader, 1):
            out = model(d)
            tempsurvival = torch.tensor([data.survival for data in d]) # 单个batch数据中的survival data
            tempcensor = torch.tensor([data.censor for data in d]) # 0 删失 1 死亡
            tempID = np.asarray([data.item for data in d]) # item='TCGA-F5-6464-01Z-00-DX1'

            tempSet, EpochSet = \
                cox_LOOV_sort(out, tempsurvival, tempcensor, tempID,
                    EpochRisk, EpochSurv, EpochCensor, EpochID)

            EpochRisk, EpochSurv, EpochCensor, EpochID = EpochSet
              
        Epochacc = accuracytest(torch.tensor(EpochSurv), torch.tensor(EpochRisk), torch.tensor(EpochCensor))
        surv_df = pd.DataFrame(list(zip(EpochID, EpochRisk, EpochSurv, EpochCensor)),
                               columns = ['SampleID', 'riskScore', 'Surv', 'Censor'])
           
        return Epochacc, surv_df