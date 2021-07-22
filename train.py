#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 12:33:30 2021

@author: amadeu
"""



import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

import gc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import csv
import gc
import torch

import torch.nn as nn
import numpy as np 

import data_preprocessing as dp


import utilss
from utilss import create_exp_dir

import models.NCNet_RR_model as model



parser = argparse.ArgumentParser("train SCINet")
parser.add_argument('--data_directory', type=str, default='/home/amadeu/Desktop/SCINet/data/traffic.txt', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--seq_size', type=int, default=10, help='sequence size') # 200 oder 1000
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--test_epochs', type=int, default=10, help='num of testing epochs')

parser.add_argument('--num_steps', type=int, default=2, help='number of iterations per epoch')
parser.add_argument('--test_num_steps', type=int, default=2, help='number of iterations per testing epoch')

parser.add_argument('--note', type=str, default='try', help='note for this run')

parser.add_argument('--num_motifs', type=int, default=100, help='number of channels') # 320
parser.add_argument('--model', type=str, default='DanQ', help='path to save the model')
parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the final model')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--report_freq', type=int, default=5, help='validation report frequency')
args = parser.parse_args()


args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utilss.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))


def main():
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # criterion = nn.BCELoss()
  
    
  train_queue, test_queue = dp.data_preprocessing(args.data_directory, args.seq_size, args.batch_size)
    
    
  net_args = {
    "rr_block": model.ResidualBlock,
    "num_blocks": [2, 2, 2, 2]
    }

  model = model.ResNet(**net_args).float().to(device)
    
  #model = model.ResNet().to(device)
  criterion = nn.MSELoss().to(device)
  optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        
   
  train_losses = []
  #valid_losses = []
  test_losses = []
 

  for epoch in range(args.epochs):
      
      
      train_start = time.strftime("%Y%m%d-%H%M")

      
      train_loss = Train(model, train_queue, optimizer, criterion, device, args.num_steps, args.report_freq)

      
      train_losses.append(train_loss)
     
      
      #if epoch % args.report_freq == 0:
          
      #    valid_loss = Valid(model, train_queue, valid_queue, optimizer, criterion, device, args.num_steps, args.report_freq)
          
      #    valid_losses.append(valid_loss)
          
        
      trainloss_file = '{}-train_loss-{}'.format(args.save, train_start)
      np.save(trainloss_file, train_losses)
      
      
      #validloss_file = '{}-valid_loss-{}'.format(args.save, train_start)
      #np.save(validloss_file, valid_losses)
      
      

  for epoch in range(args.test_epochs): 
      
      test_loss = Valid(model, test_queue, optimizer, criterion, device, args.test_num_steps, args.report_freq)
      
      test_losses.append(train_loss)
      
  testloss_file = '{}-test_loss-{}'.format(args.save, train_start)
  np.save(testloss_file, test_losses)
      


      
    

# train_loader, num_steps = train_queue, 2

def Train(model, train_loader, optimizer, criterion, device, num_steps, report_freq):
        
    objs = utilss.AvgrageMeter()
    
    total_loss = 0
    start_time = time.time()
    
    for idx, (inputs, targets) in enumerate(train_loader):
        
        if idx > num_steps:
            break
        
        input, y_true = inputs, targets
        
        input = input.reshape(2, 1, args.seq_size)
       
        model.train()
        #input = input.transpose(1, 2).float()

        input = input.to(device)#.cuda()
        
        #batch_size = input.size(0)
    
        optimizer.zero_grad()
         
        y_pred = model(input.float()) 

        loss = criterion(y_pred.float(), y_true.float())

        loss.backward()
        optimizer.step()
       
        objs.update(loss.data, args.batch_size) # calculate running average of loss
       
      
    return objs.avg
    

def Valid(model, valid_loader, optimizer, criterion, device, num_steps, report_freq):
   
    objs = utilss.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        
        for idx, (inputs, labels) in enumerate(valid_loader):
            
            
            if idx > num_steps:
                break
        
            input, y_true = inputs, labels
            input = input.reshape(2, 1, args.seq_size)

            input = input.to(device)#.cuda()  
            
            y_pred = model(input.float()) #, (state_h, state_c))


            loss = criterion(y_pred.float(), y_true.float())
                
            objs.update(loss.data, args.batch_size)
       
    return objs.avg #top1.avg, objs.avg


if __name__ == '__main__':
  main() 