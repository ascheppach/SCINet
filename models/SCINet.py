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
        
        
    def realign(self): # realign elements to reverse the odd even splitting
        
        def pre_realign(v, combination):
            # v a list, combination a binary list
            # in each step v is reduced by eliminating elements of v with odd or even indices, depending on elements of binary list 'combination'
            for i in range(len(combination)):
                if combination[i] == 0 :
                    # reduce v to those elements of v with even index
                    v = v[::2]
                else :
                    # reduce v to those elements of v with odd index
                    v = v[1::2]
            return v 
        
        ve = list(range(self.seq_size)) # list of [0, 1, ... ,(n-1)]
        output = list()
        a = np.array(list(range(2))) 
        points = product(a, repeat = self.L) # generates a matrix of all possible k-length combinations of 0 and 1 
        mat = (list(points))
        
        for j in range((2**self.L)): # for each reduced component of 've'
            output.extend(pre_realign(v = ve, combination = mat[j]))
             #reduce ve by iteratively eliminating elements with odd or even indices according to the elements of the next row of 'mat', then append the reduced vector to 'output'

        reverse_idx = torch.Tensor(output).to(device)
        reverse_idx = torch.Tensor(reverse_idx).long()
        a = np.row_stack([reverse_idx.tolist(), list(range(self.seq_size))])
        a = a[:, a[0, :].argsort()]
        
        return a[1,:]
        
        
    def forward(self, x):
        
        residual = x
        F_01 = x
        for l in range(1, self.L + 1):
            i = 1
            j = 1
            #print(" ")
            while i <= 2**l:
                exec("F_" + str(l) + str(i) + ", F_" + str(l) + str(i+1) +
                      " = self.sci_level_" 
                      + str(l-1) + "(F_" + str(l-1) + str(j) + ")")
                i += 2
                j += 1
        
        F_concat = F_01
        exec("F_concat = F_" + str(self.L) + "1")  
        for i in range(2, 2**self.L + 1):
            exec("F_concat = torch.cat([F_concat, F_"+ str(self.L) + str(i) + "], dim = 2)")   
        
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

        
# Level 1
sci_level_0 = SCI_Block(1,16, 3, 1, 2, split, 20)
F_01 = sci_level_0(x) 
# Level 2
sci_level_1 = SCI_Block(1, 16, 3, 1, 2, split, 10)
F_21 = sci_level_1(F_01[0]) 
F_22 = sci_level_1(F_01[1]) 
# Level 3
sci_level_2 = SCI_Block(1, 16, 3, 1, 2, split, 20)
sci_level_3 = SCI_Block(1, 16, 3, 1, 2, split, 10)

L = 2
residual = x
F_01 = x
for l in range(1, L + 1):
    i = 1
    j = 1
    #print(" ")
    while i <= 2**l:
        exec("F_" + str(l) + str(i) + ", F_" + str(l) + str(i+1) +
              " = sci_level_" 
              + str(l-1) + "(F_" + str(l-1) + str(j) + ")")
        i += 2
        j += 1

F_concat = F_01
exec("F_concat = F_" + str(L) + "1")   
for i in range(2, 2**L + 1):
    exec("F_concat = torch.cat([F_concat, F_"+ str(L) + str(i) + "], dim = 2)")   

