#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jul 22 08:07:31 2021

@author: amadeu
"""



import csv

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import Counter
from argparse import Namespace
import pandas as pd

import os

from numpy import array
import torch
import gc
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import Dataset,DataLoader
import numpy as np


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



#conv = conv_op(1, 16, 2, 1, 5)
#x = conv(x)




class SCI_Block(nn.Module): # 
    def __init__(self, in_channels, expand, kernel, stride, padding, split, seq_size):
        super(SCI_Block, self).__init__()
        
        self.nu = conv_op(in_channels, expand, kernel, stride, padding)
        self.psi = conv_op(in_channels, expand, kernel, stride, padding)
        self.ro = conv_op(in_channels, expand, kernel, stride, padding)
        self.phi = conv_op(in_channels, expand, kernel, stride, padding)
        
        self.idx_even, self.idx_odd = split(seq_size)
        

    def forward(self, x):
        #residual = x
        F_even = x[:,:,self.idx_even]
        F_odd = x[:,:,self.idx_odd]
        
        # print(F_even.shape)
        # print(torch.exp(self.psi(F_odd)).shape)

        F_even_s = F_even * torch.exp(self.psi(F_odd))
        F_odd_s = F_odd * torch.exp(self.phi(F_even))

        F_even_final = F_odd_s - self.nu(F_odd_s) # in the paper, they write you can add or subtract, but in the picture
        # on page 4 they do an addition for F_odd and subtraction for F_even
        F_odd_final = F_even_s + self.ro(F_even_s)       

       
        return F_even_final, F_odd_final
    
    


x = torch.rand(2,1,20) # input with batch_size 2 and sequence length 20


# Level 1
level_1_sci = SCI_Block(1,16, 3, 1, 2, split, 20)
F_even_1, F_odd_1 = level_1_sci(x) 
# Level 2
level_2_sci = SCI_Block(1, 16, 3, 1, 2, split, 10)
F_even_2, F_odd_2 = level_2_sci(F_even_1) 
F_even_2, F_odd_2 = level_2_sci(F_odd_1) 


# in_channels, expand, kernel, stride, padding, split, seq_sizes = 1, 16, 3, 1, 2, split, [20, 10]




class SCI_Net(nn.Module): # 
    def __init__(self, in_channels, expand, kernel, stride, padding, split, seq_sizes, SCI_Block):
        super(SCI_Net, self).__init__()
        
        self.level_1_sci = SCI_Block(in_channels, expand, kernel, stride, padding, split, seq_sizes[0])
        self.level_2_sci = SCI_Block(in_channels, expand, kernel, stride, padding, split, seq_sizes[1])
        
        self.fc = nn.Linear(20*1,20) #because 10 neurons/seq_len are now 8 neurons

        
    #    self.idx_even, self.idx_odd = split(seq_size)
        
        
    def realign():
        bl = np.array([0,10,5,15])
        reverse_idx = []
        reverse_idx.append(bl)
        for i in range(4):
            bl = bl + 1
            reverse_idx.append(bl)
            
        reverse_idx = np.concatenate(reverse_idx)
        reverse_idx = torch.Tensor(reverse_idx).to(device)
        reverse_idx = torch.Tensor(reverse_idx).long()
    
        
        return reverse_idx
    
        self.reverse_idx = realign()
        


    def forward(self, x):
        
        residual = x # [2,1,20]
        
        ## Level 1
        F_even_1, F_odd_1  = self.level_1_sci(x)
        # F_even_1, F_odd_1 = level_1_sci(x) 
        
        ## Level 2
        #level_2_sci = SCI_Block(1, 16, 3, 1, 2, split, 10)
        F_even_21, F_odd_22 = self.level_2_sci(F_even_1) 
        
        F_even_23, F_odd_24 = self.level_2_sci(F_odd_1) 
        
        F_concat = torch.cat([F_even_21, F_odd_22, F_even_23, F_odd_24], dim=2)
        
        reverse_idx = realign()
        
        F_concat = F_concat[:,:,reverse_idx.long()]
        
        F_concat += residual
        
        output = self.fc(F_concat)

       
        return output







seq_sizes = [20, 10]

sci_net = SCI_Net(1, 16, 3, 1, 2, split, seq_sizes, SCI_Block)

X_k = sci_net(x)








