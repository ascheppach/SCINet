# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 13:27:31 2021

@author: vszab
"""

import numpy as np
import torch
import torch.nn as nn
import torch.utils

import utilss
import data_preprocessing as dp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# evaluate them with the RSE-loss and Correlation metric from the paper
class RSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predict, target):
        return torch.sqrt((predict-target).pow(2).sum())/torch.sqrt((target-target.mean()).pow(2).sum())
    
class Correlation(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predict, target):
        target_dev = target-target.mean()
        predict_dev = predict-predict.mean()
        return (target_dev*predict_dev).sum()/torch.sqrt((target_dev.pow(2)*predict_dev.pow(2)).sum())


# look at the losses and choose best model
data = np.load('output2021_08_25/EXPsearch-try-20210825-180827-val_loss-20210825-1920.npy', allow_pickle=True)

# load test data
train_queue, val_queue, test_queue = dp.data_preprocessing('data/traffic.txt', 168, 16, 3)

# load best model
model = torch.load("output2021_08_25/run1.pth", map_location=torch.device('cpu'))

# make predictions with the model and evaluate
objs = utilss.AvgrageMeter()
RSE_losses = []
Corr_metrics = []

for idx, (inputs, targets) in enumerate(test_queue):
    input, y_true = inputs, targets
    # reshape for consistent size in calculation
    input = input.reshape(16, 1, 168)
    y_true = y_true.reshape(16, 1, 3)
    
    input = input.to(device)
    y_true = y_true.to(device)
    
    # predict, calculate loss
    model.eval()
    y_pred = model(input.float()) 
    criterion1 = RSELoss().to(device)
    criterion2 = Correlation().to(device)
    
    RSE = criterion1(y_pred.float(), y_true.float())
    Corr = criterion2(y_pred.float(), y_true.float())
    
    #print('RSE loss is {}'.format(RSE))
    #print('Correlation is {}'.format(Corr))
    
    objs.update(RSE.data, 16)
    RSE_loss = objs.avg
    RSE_losses.append(RSE_loss)
    Corr_metric = objs.avg
    objs.update(Corr.data, 16)
    Corr_metrics.append(Corr_metric)



