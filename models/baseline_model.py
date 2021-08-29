#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 11:47:02 2021

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


class baseline_CNN(nn.Module):
    def __init__(self):
        super(baseline_CNN,self).__init__()
        self.conv1d = nn.Conv1d(1,64,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(1280,50)
        self.fc2 = nn.Linear(50,3)
        
    def forward(self,x):
        #print(x.shape)
        x = self.conv1d(x)
        #print(x.shape)
        x = self.relu(x)
        # x = x.view(-1)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x