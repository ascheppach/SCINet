#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 20:03:06 2021

@author: amadeu
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
    return idx_even.to(device), idx_odd.to(device) 


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

# even, odd = split(seq_size)
# seq_size = 168
# x = torch.rand(2,862,168)

# in_channels, expand, kernel, stride, padding = 862, 2, 5, 1, 4

class SCI_Block(nn.Module): # 
    def __init__(self, in_channels, expand, kernel, stride, padding, split, seq_size, batch_size):
        super(SCI_Block, self).__init__()
        
        # convolutional layer for each operation
        self.nu = conv_op(in_channels, expand, kernel, stride, padding)
        self.psi = conv_op(in_channels, expand, kernel, stride, padding)
        self.ro = conv_op(in_channels, expand, kernel, stride, padding)
        self.phi = conv_op(in_channels, expand, kernel, stride, padding)
        self.idx_even, self.idx_odd = split(seq_size)
        # idx_even, idx_odd = split(seq_size)
        
    def forward(self, x):
        # split sequence to even/odd block
      
        F_even = x[:,:,self.idx_even]
        F_odd = x[:,:,self.idx_odd]
        # print(torch.exp(self.psi(F_odd)).size())
        # print(torch.exp(self.phi(F_even)).size())
        F_even_s = F_even * torch.exp(self.psi(F_odd))
        F_odd_s = F_odd * torch.exp(self.phi(F_even))

        F_even_final = F_odd_s - self.nu(F_odd_s) # in the paper, they write you can add or subtract, but in the picture
        # on page 4 they do an addition for F_odd and subtraction for F_even
        F_odd_final = F_even_s + self.ro(F_even_s)       

        return F_even_final.to(device), F_odd_final.to(device)


# in_channels, expand, kernel, stride, padding, split, seq_size, batch_size = 1, 2, 5, 1, 4, split, 168, 2
class SCI_Net(nn.Module): # 

    def __init__(self, in_channels, expand, kernel, stride, padding, split, seq_size, batch_size, SCI_Block, L, horizon):
        
        super(SCI_Net, self).__init__()
        
        self.block_list = nn.ModuleList()
        
        inc=1
        offset = 0
        
        for i in range(L): 
            sub_block = SCI_Block(in_channels, expand, kernel, stride, padding, split, seq_size/(1*inc), batch_size)
            for j in range(inc):
                self.block_list.add_module(str(offset+j), sub_block)
            inc *= 2
            offset = len(self.block_list)

        self.fc1 = nn.Linear(seq_size, 50).to(device) 
        
        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(50, horizon).to(device) 
        self.L = L
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.horizon = horizon
        
        
    def realign(self): # realign elements to reverse the odd-even splitting
        
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
        reverse_idx = reverse_idx.long()
        a = np.row_stack([reverse_idx.tolist(), list(range(self.seq_size))])
        a = a[:, a[0, :].argsort()]
        
        return a[1,:]
        
        
    def forward(self, x):
        
        residual = x # residual connection in SCInet
        
        prev_F = [x]
        start=0
        end=1
        for i in range(self.L):
            prev_res = []
            for j, sci_block in enumerate(self.block_list[start:end]):
                res = sci_block(prev_F[j])
                #results.append((F_o, F_e))
                for k in res:
                    prev_res.append(k) # with i=0 we have list with 2 elements; with i=1 we have list with 4 Elements, with i=2 list with 8 elements and so on ..
            prev_F = prev_res # 1 element, 2 elements, 4 elements, 8 elements
            start = end # 0, 1, 3
            end = (start*2) + 1 # 1, 3, 7 ; because in each SCINet, the number of SCI_Blocks get doubled

        # After concatenation, the odd-even splitting order is reversed
        reverse_idx = self.realign()
        
        F_concat = torch.cat((prev_F), dim=2)

        F_concat = F_concat[:,:,reverse_idx]
        F_concat += residual
        # Finally a fully connected layer is used to get the output
        # F_concat = torch.flatten(F_concat, 1)
        
        x = self.fc1(F_concat)
        
        x = self.relu(x)
                
        output = self.fc2(x)
        
        return output
