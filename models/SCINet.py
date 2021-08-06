#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jul 22 08:07:31 2021

@author: Scheppach Amadeu, Szabo Viktoria, To Xiao-Yin
"""

from argparse import Namespace
from collections import Counter
import csv
import gc
from itertools import product
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from numpy import array
import os
import pandas as pd

from tqdm import tqdm_notebook as tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# split into even and uneven
# indexing for F_even
def split(seq_size):
    idx_odd = torch.Tensor(np.arange(0,seq_size,2)).to(device) # because 1th, 3th... element ist 0th, 2th ... in python
    idx_odd = idx_odd.long()
    
    # indexing for F_odd
    idx_even = torch.Tensor(np.arange(1,seq_size,2)).to(device)
    idx_even = idx_even.long()
    
    # create F_even and F_odd
    
    return idx_even, idx_odd


def conv_op(in_channels, expand, kernel, stride, padding): # 
    convs = nn.Sequential(nn.ReplicationPad1d(padding),
                          nn.Conv1d(in_channels, in_channels*expand, kernel_size=kernel, 
                          stride=stride, bias=False),
                          nn.LeakyReLU(negative_slope=0.01),
                          nn.Dropout(0.5),
                          nn.Conv1d(in_channels*expand, in_channels, kernel_size=kernel, 
                          stride=stride, bias=False),
                          nn.Tanh())
    return convs


class SCI_Block(nn.Module): # 
    def __init__(self, in_channels, expand, kernel, stride, padding, split, seq_size):
        super(SCI_Block, self).__init__()
        
        # convolutional layer for each operation
        self.nu = conv_op(in_channels, expand, kernel, stride, padding)
        self.psi = conv_op(in_channels, expand, kernel, stride, padding)
        self.ro = conv_op(in_channels, expand, kernel, stride, padding)
        self.phi = conv_op(in_channels, expand, kernel, stride, padding)
        
        self.idx_even, self.idx_odd = split(seq_size)

    def forward(self, x):
        # split sequence to even/odd block
        F_even = x[:,:,self.idx_even]
        F_odd = x[:,:,self.idx_odd]
        F_even_s = F_even * torch.exp(self.psi(F_odd))
        F_odd_s = F_odd * torch.exp(self.phi(F_even))

        F_even_final = F_odd_s - self.nu(F_odd_s) # in the paper, they write you can add or subtract, but in the picture
        # on page 4 they do an addition for F_odd and subtraction for F_even
        F_odd_final = F_even_s + self.ro(F_even_s)       

        return F_even_final, F_odd_final


class SCI_Net(nn.Module): # 
    def __init__(self, in_channels, expand, kernel, stride, padding, split, seq_size, SCI_Block, L):
        super(SCI_Net, self).__init__()
        for i in range(L):
            exec("self.sci_level_" + str(i) + " = SCI_Block(in_channels, expand, kernel, stride, padding, split," + str(int(seq_size)/(2**i)) + ")")
        self.fc = nn.Linear(seq_size*1,1) #because 10 neurons/seq_len are now 8 neurons
        self.L = L
        self.seq_size = seq_size
        
    def realign(self):
        def pre_realign(v,combination):
            for i in range(len(combination)):
                if combination[i]==0 :
                    v=v[::2]
                else :
                    v=v[1::2]
            return v
        
        ve=list(range(self.seq_size))
        output=list()
        a=np.array(list(range(2)))
        points=product(a,repeat=self.L)
        mat=(list(points))
        for j in range((2**self.L)):
            output.extend(pre_realign(v=ve,combination=mat[j]))
        reverse_idx = torch.Tensor(output).to(device)
        reverse_idx = torch.Tensor(reverse_idx).long()
        a = np.row_stack([reverse_idx.tolist(), list(range(self.seq_size))])
        a = a[:, a[0, :].argsort()]
        return a[1,:]
        

############ NEEDS TO BE SOFTCODED
    def forward(self, x):
        
        residual = x # [2,1,20]
        
        ## Level 1
        F_even_1, F_odd_1  = self.sci_level_0(x)
        
        ## Level 2
        F_even_21, F_odd_22 = self.sci_level_1(F_even_1) 
        
        F_even_23, F_odd_24 = self.sci_level_1(F_odd_1) 
        
        F_concat = torch.cat([F_even_21, F_odd_22, F_even_23, F_odd_24], dim=2)
        reverse_idx = self.realign()
        F_concat = F_concat[:,:,reverse_idx]
        F_concat += residual
        
        output = self.fc(F_concat)

       
        return output

class stackedSCI(nn.Module):
    def __init__(self, in_channels, expand, kernel, stride, padding, split, seq_size, SCI_Block, K, L):
        super(stackedSCI, self).__init__()
        self.K = K
        # Create SCINet block layers for each K
        for i in range(K):
            exec("self.sci_stacK_" + str(i) + 
                 "= SCI_Net(in_channels, expand, kernel, stride, padding, split, seq_size, SCI_Block, L)")

    def forward(self, x):
        x_0 = x
        X = list()
        # stack K SCINets
        for j in range(self.K):
            # Save each SCINet in list X
            exec("X" "= X.append(self.sci_stacK_" + str(j) + "(x_" + str(j) + "))")
            # Save prediction in x_(j+1) in order to create Tensor containing all predictions for loss computation
            exec("x_" + str(j+1) + " = torch.cat((x_" + str(j) + ", X[" + str(j) + "]),2)[:,:,1:21]")
        # reshape the output to match with true values
        out = torch.stack(X).reshape(-1,self.K,1)
        return out


## Testrun
x = torch.rand(2,1,20) # input with batch_size 2 and sequence length 20

# Level 1
level_1_sci = SCI_Block(1,16, 3, 1, 2, split, 20)
F_even_1, F_odd_1 = level_1_sci(x) 
# Level 2
level_2_sci = SCI_Block(1, 16, 3, 1, 2, split, 10)
F_even_2, F_odd_2 = level_2_sci(F_even_1) 
F_even_2, F_odd_2 = level_2_sci(F_odd_1) 

sci_net = SCI_Net(1, 16, 3, 1, 2, split, 20, SCI_Block, 2)

X_k = sci_net(x)

stacki = stackedSCI(in_channels = 1, 
                    expand = 16, 
                    kernel = 2, 
                    stride = 1, 
                    padding = 1, 
                    split = split, 
                    seq_size = 20, 
                    SCI_Block = SCI_Block, 
                    K = 2, 
                    L = 2).float().to(device)

XK = stacki(x)




