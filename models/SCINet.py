#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 10:18:28 2021

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


class SCI_Block(nn.Module): # 
    def __init__(self, in_channels, expand, kernel, stride, padding, split, seq_size, batch_size):
        super(SCI_Block, self).__init__()
        
        # convolutional layer for each operation
        self.nu = conv_op(in_channels, expand, kernel, stride, padding)
        self.psi = conv_op(in_channels, expand, kernel, stride, padding)
        self.ro = conv_op(in_channels, expand, kernel, stride, padding)
        self.phi = conv_op(in_channels, expand, kernel, stride, padding)
        self.idx_even, self.idx_odd = split(seq_size)
        
    def forward(self, x):
        # split sequence to even/odd block
        # print(x.size())
        # print(self.idx_even)
        # print(self.idx_odd)
        F_even = x[:,:,self.idx_even]
        F_odd = x[:,:,self.idx_odd]
        # print(torch.exp(self.psi(F_odd)).size())
        # print(torch.exp(self.phi(F_even)).size())
        F_even_s = F_even * torch.exp(self.psi(F_odd))
        F_odd_s = F_odd * torch.exp(self.phi(F_even))

        # print(F_odd_s.size())
        # print(F_even_s.size())
        # print(self.nu(F_odd_s).size())
        # print(self.ro(F_even_s).size())
        F_even_final = F_odd_s - self.nu(F_odd_s) # in the paper, they write you can add or subtract, but in the picture
        # on page 4 they do an addition for F_odd and subtraction for F_even
        F_odd_final = F_even_s + self.ro(F_even_s)       

        return F_even_final.to(device), F_odd_final.to(device)


class SCI_Net(nn.Module): # 
    def __init__(self, in_channels, expand, kernel, stride, padding, split, seq_size, batch_size, SCI_Block, L, horizon):
        super(SCI_Net, self).__init__()
        for i in range(L):
            # print(str(seq_size),str(2**i))
            #print("self.sci_level_" + str(i) + " = SCI_Block(in_channels, expand, kernel, stride, padding, split," + str(int(seq_size)/(2**i)) + ")")
            exec("self.sci_level_" + str(i) + " = SCI_Block(in_channels, expand, kernel, stride, padding, split," + str(int(seq_size)/(2**i)) + ", batch_size)")
        self.fc = nn.Linear(seq_size*1, horizon).to(device) #because 10 neurons/seq_len are now 8 neurons
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
        F_01 = x 
        exec("F_" + str(self.L) + "1 = list()")
        # Use previously defined SCI-Blocks (sci_level_...) on every level of SCInet
        # Output of a SCI-Block are two tensors (even and odd)
        # In the next level the next SCI-Block is used on both of those tensors and so on, until level L is reached
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
                
        # All outputs of the last level have to be concatenated
        F_concat = F_01
        F_concat = eval("F_" + str(self.L) + "1")
        
        for i in range(2, 2**self.L + 1):
            F_concat = eval("torch.cat([F_concat, F_"+ str(self.L) + str(i) + "], dim = 2)")       
        
        # After concatenation, the odd-even splitting order is reversed
        reverse_idx = self.realign()
        F_concat = F_concat[:,:,reverse_idx]
        F_concat += residual
        # print("F_concat before")
        # print(F_concat.size())
        # print(F_concat)
        
        # Finally a fully connected layer is used to get the output
        F_concat = torch.flatten(F_concat, 1)
        # print("F_concat after")
        # print(F_concat.size())
        # print(F_concat)
        
        output = self.fc(F_concat)
        # print("Output fc")
        # print(output.size())
        # print(output)
        
        output = output.reshape(self.batch_size, 1, self.horizon)
        # print("Output reshape:")
        # print(output.size())
        # print(output)
        
        return output


class stackedSCI(nn.Module):
    def __init__(self, in_channels, expand, kernel, stride, padding, split, seq_size, batch_size, SCI_Block, K, L, horizon):
        super(stackedSCI, self).__init__()
        self.K = K
        self.seq_size = seq_size
        self.horizon = horizon
        # Create SCINet block layers for each K
        for i in range(K):
            exec("self.sci_stacK_" + str(i) + 
                 "= SCI_Net(in_channels, expand, kernel, stride, padding, split, seq_size, batch_size, SCI_Block, L, horizon)")
                    #(self, in_channels, expand, kernel, stride, padding, split, seq_size, SCI_Block, L)
    def forward(self, x):
        x_0 = x
        # print("x_0")
        # print(x_0)
        X = list()
        # stack K SCINets
        for j in range(self.K):
            #print("STACK " + str(j))
            # Save each SCINet in list X
            #print("X" + " = X.append(self.sci_stacK_" + str(j) + "(x_" + str(j) + "))")
            #print("x_" + str(j+1) + " = torch.cat((x_" + str(j) + ", X[" + str(j) + "]),2)[:,:,1:" + str(self.seq_size+1) +"]")
            #print(eval("x_" + str(j)))
            exec("X" + " = X.append(self.sci_stacK_" + str(j) + "(x_" + str(j) + "))")
            # print("X nach append")
            # print(X)
            # Save prediction in x_(j+1) in order to create Tensor containing all predictions for loss computation
            # print(eval("torch.cat((x_" + str(j) + ", X[" + str(j) + "]),2)"))
            # print("x_" + str(j) + " before:")
            # exec("print(x_" + str(j) + ".size())")
            # exec("print(x_" + str(j) + ")")
            
            # print("X[" + str(j) + "]")
            # exec("print(X[" + str(j) + "].size())")
            # exec("print(X[" + str(j) + "])")            
            
            # print("x_" + str(j+1) + " = torch.cat((x_" + str(j) + ", X[" + str(j) + "]),2)[:,:,1:" + str(self.seq_size+1) +"]")
            exec("x_" + str(j+1) + " = torch.cat((x_" + str(j) + ", X[" + str(j) + "]),2)[:,:," + str(self.horizon) + ":" + str(self.seq_size + self.horizon) +"]")
            
        #     print("x_" + str(j+1) + " after:")
        #     exec("print(x_" + str(j+1) + ".size())")
        #     exec("print(x_" + str(j+1) + ")")
        # # reshape the output to match with true values
        # # out = torch.stack(X).reshape(-1,self.K,1)
        # print("stacked X:")
        # #print(X.size())
        # print(X)
        
        #out = torch.stack(X).reshape(-1, self.K, self.horizon)
        out = torch.cat(X, 1)
        # print("stacki output:")
        # print(out.size())
        # print(out)
        return out


## Testrun
# x = torch.rand(2,1,40) # input with batch_size 2 and sequence length 40

# sci_net = SCI_Net(1, 16, 3, 1, 2, split, 40, 2, SCI_Block, 3, 3)
# X_k = sci_net(x)
# stacki = stackedSCI(in_channels = 1, 
#                     expand = 10,
#                     kernel = 5,
#                     stride = 1,
#                     padding = 4,
#                     split = split,
#                     seq_size = 40,
#                     batch_size = 2,
#                     SCI_Block = SCI_Block,
#                     K = 4,
#                     L = 3,
#                     horizon = 3).float().to(device)

# XK = stacki(x)
# XK
# blubb = torch.cat(XK,1)
# blubb
# K = 4
# horizon = 3
# XK.reshape(-1, K, horizon)

# Level 1
# sci_level_0 = SCI_Block(1,16, 3, 1, 2, split, 40)
# F_01 = sci_level_0(x) 
# # Level 2
# sci_level_1 = SCI_Block(1, 16, 3, 1, 2, split, 20)
# F_21 = sci_level_1(F_01[0]) 
# F_22 = sci_level_1(F_01[1]) 
# # Level 3
# sci_level_2 = SCI_Block(1, 16, 3, 1, 2, split, 10)
# sci_level_3 = SCI_Block(1, 16, 3, 1, 2, split, 10)

# L = 3
# residual = x
# F_01 = x
# for l in range(1, L + 1):
#     i = 1
#     j = 1
#     print(" ")
#     while i <= 2**l:
#         #print("F_" + str(l) + str(i) + ", F_" + str(l) + str(i+1) +
#               " = sci_level_" 
#               + str(l-1) + "(F_" + str(l-1) + str(j) + ")")
#         exec("F_" + str(l) + str(i) + ", F_" + str(l) + str(i+1) +
#               " = sci_level_" 
#               + str(l-1) + "(F_" + str(l-1) + str(j) + ")")
#         i += 2
#         j += 1

# F_concat = F_01
# exec("F_concat = F_" + str(L) + "1")   
# for i in range(2, 2**L + 1):
#     exec("F_concat = torch.cat([F_concat, F_"+ str(L) + str(i) + "], dim = 2)")   
# F_concat