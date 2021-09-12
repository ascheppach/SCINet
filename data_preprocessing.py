#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Scheppach Amadeu, Szabo Viktoria, To Xiao-Yin
"""


######## data preprocessing #####
import torch
from collections import Counter
import numpy as np
from numpy import array
from torch.utils.data import Dataset,DataLoader

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

import math



        
def data_preprocessing(data_file, seq_size, batch_size, K, name):
    
    
    # depending on which dataset is used, the preprocessing is done differently
    def open_data_traffic(data_file):
        with open(data_file, 'r') as f:
            text = f.read()
            
        text = text.split("\n") # there are 862 timesteps after each \n 
        # so we have 17544 timesteps, and each of these elements has 862 features
        text = text[:-1]
        
        sample = text[0].split(",")
        sample = np.array(sample)
        sample = sample.astype(float)
        samples = np.reshape(sample, (862,1))

        #all_samples = []
        for sample in text: # each samples consists of 862 time steps
            # sample = text [1]
            sample = sample.split(",")
            sample = np.array(sample)
            sample = sample.astype(float)
            sample = np.reshape(sample, (862,1))
            samples = np.append(samples, sample, axis=1)
            
        #all_samples = np.concatenate(all_samples, axis=0)
            
        return samples
    
        
    def open_data_elec(data_file):
        with open(data_file, 'r') as f:
            text = f.read()
            
            text = text.split("\n") 
            # don't use headers, which are the client codenames
            text = text[1:]
            
            all_samples = []
            for sample in text:
                sample = sample.split(";")
                sample = np.array(sample)
                all_samples.append(sample)
                
            all_samples_new = []
            for i in all_samples:
                # delete first observation (date and time)
                i = np.delete(i, 0)
                for j in range(len(i)):
                    i[j] = float(i[j].replace(',','.'))
                i = np.array(i)
                i = i.astype(float)
                all_samples_new.append(i)
        
        return all_samples_new

    # create sequences based on the look-back horizon (seq_size) and prediction horizon (K)
    def create_sequences(all_samples, seq_size, K): 
        x = list()
        y = list()
        
        #data_len = len(all_samples)
        #train_len = math.floor(0.6*data_len)
        #valid_len = math.floor(0.2*data_len)
        #test_len = math.floor(0.2*data_len)
        
        for i in range(len(all_samples)): # 0:3 für feat, 3:6 für target
            # i=4
            # seq_size=3
            idx = i + seq_size # 0+3
            
            if (idx+K) > len(all_samples[1])-1: 
                break
            
            # add K positions to label to predict the K next timesteps
            feat_seq, target_seq = all_samples[:,i:idx], all_samples[:,idx:(idx+K)] # target labels for CNN
            x.append(feat_seq)
            y.append(target_seq)
            
        return torch.Tensor(x), torch.Tensor(y)


    class get_data(Dataset):
        def __init__(self,feature,target):
            self.feature = feature
            self.target = target
        def __len__(self):
            return len(self.feature)
        def __getitem__(self,idx):
            item = self.feature[idx]
            label = self.target[idx]
            return item,label
    
    if name=='traffic':
        all_samples = open_data_traffic(data_file)
    
    if name=='electricity':
        all_samples = open_data_elec(data_file)
    
    x, y = create_sequences(all_samples, seq_size, K)
    
    # Split in train, validation and test data
    rest_feat, test_feat, rest_targ, test_targ = train_test_split(
            x, y, test_size=0.2) # test data = 20%
    
    # validation = 10% of complete data and therefore 12,5% of the 80% rest data
    train_feat, valid_feat, train_targ, valid_targ = train_test_split(
            rest_feat, rest_targ, test_size=0.125) 
    
    train = get_data(train_feat, train_targ)
    valid = get_data(valid_feat, valid_targ)
    test = get_data(test_feat, test_targ)

    
    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)#  shuffle ensures random choices of the sequences
    valid_loader = torch.utils.data.DataLoader(valid, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size, shuffle=False)

  
    return train_loader, valid_loader, test_loader