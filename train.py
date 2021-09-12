#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: @author: Scheppach Amadeu, Szabo Viktoria, To Xiao-Yin
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

# import own functions
import data_preprocessing as dp
import utilss
from utilss import create_exp_dir


parser = argparse.ArgumentParser("train SCINet")
# Data and save information
parser.add_argument('--data_directory', type=str, default='data/traffic.txt', help='location of the data corpus')
parser.add_argument('--model', type=str, default='SCINet', help='path to save the model')
parser.add_argument('--data_name', type=str, default='traffic', help='name of data used')
parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the final model')
# Specification of Training 
parser.add_argument('--horizon', type=int, default=3, help='prediction horizon') 
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs') 
parser.add_argument('--val_epochs', type=int, default=10, help='num of validation epochs')
parser.add_argument('--num_steps', type=int, default=2, help='number of iterations per epoch')
parser.add_argument('--val_num_steps', type=int, default=2, help='number of iterations per valing epoch')
parser.add_argument('--report_freq', type=int, default=5, help='validation report frequency') # 5 in paper
# General hyperparameters
parser.add_argument('--batch_size', type=int, default=2, help='batch size') 
parser.add_argument('--seq_size', type=int, default=168, help='sequence size')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='init learning rate') 
parser.add_argument('--k', type=int, default=5, help='kernel size')
parser.add_argument('--stride', type=int, default=1, help='stride')
parser.add_argument('--padding', type=int, default=4, help='padding')
parser.add_argument('--num_motifs', type=int, default=100, help='number of channels')
# Paper specific hyperparameters
parser.add_argument('--h', type=int, default=2, help='extention of input channel')
parser.add_argument('--K', type=int, default=1, help='number of stacks')
parser.add_argument('--L', type=int, default=3, help='Number of SCI-Block levels')
# Other
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--seed', type=int, default=4321, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

parser.add_argument('--model_path', type=str,  default='./run1.pth',
                    help='path to save the trained model')

args = parser.parse_args()
args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))

# TRAINING LOOP
def Train(model, train_loader, optimizer, criterion, device, num_steps, report_freq, horizon):
    
    # Initialize params
    objs = utilss.AvgrageMeter()
    total_loss = 0
    start_time = time.time()
    
    # run *num_steps* training steps 
    # train_loader, valid_loader = train_queue, val_queue
    # num_steps=2
    for idx, (inputs, targets) in enumerate(train_loader):
        if idx > num_steps:
            break
        # define input and targets for step
        input, y_true = inputs, targets
        # reshape for consistent size in calculation
        # input = input.reshape(args.batch_size, 1, args.seq_size)
        # y_true = y_true.reshape(args.batch_size, 1, args.horizon)
        
        # train the model
        model.train()
        input = input.to(device)
        y_true = y_true.to(device)
        
        # predict, calculate loss, update weights
        optimizer.zero_grad()
        y_pred = model(input.float()) 
        loss = criterion(y_pred.float(), y_true.float())
        loss.backward(retain_graph=True)
        optimizer.step()
        objs.update(loss.data, args.batch_size) # calculate running average of loss
       
    return objs.avg

# VALIDATION LOOP
def Valid(model, valid_loader, optimizer, criterion, device, num_steps, report_freq, horizon):
   
    # Initialize params
    objs = utilss.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        # run *val_num_steps* validation steps 
        for idx, (inputs, labels) in enumerate(valid_loader):
            if idx > num_steps:
                break            
            # define input and targets for step
            input, y_true = inputs, labels
            # reshape for consistent size in calculation
            #input = input.reshape(args.batch_size, 1, args.seq_size)
            #y_true = y_true.reshape(args.batch_size, 1, args.horizon)

            # predict, calculate loss, save value
            input = input.to(device)
            y_true = y_true.to(device)

            y_pred = model(input.float())
            loss = criterion(y_pred.float(), y_true.float())
            objs.update(loss.data, args.batch_size)
    return objs.avg 



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global model
    train_queue, val_queue, test_queue = dp.data_preprocessing(args.data_directory, args.seq_size, args.batch_size, args.horizon, args.data_name)
        
    if args.model=='SCINet_WS':
        import models.SCINet as model
        
        # define hyperparameters
        net_args = {
            "in_channels" : 862, # because traffic data one dimensional 
            "expand" : args.h, 
            "kernel" : args.k, 
            "stride" : args.stride, 
            "padding" : args.padding, 
            "split" : model.split, 
            "seq_size" : args.seq_size, 
            "batch_size" : args.batch_size,
            "SCI_Block" : model.SCI_Block, 
            "K" : args.K, 
            "L" : args.L,
            "horizon" : args.horizon
            }
        # define model, loss, and optimizer
        model = model.stackedSCI(**net_args).float().to(device)

        
    if args.model == 'SCINet':
        
        #global model
        
        #import models.SCINet as model
        import models.SCINet2 as model
        
        net_args = {
            "in_channels" : 862, # because traffic data has 862 channels/features 
            "expand" : args.h, 
            "kernel" : args.k, 
            "stride" : args.stride, 
            "padding" : args.padding, 
            "split" : model.split, 
            "seq_size" : args.seq_size, 
            "batch_size" : args.batch_size,
            "SCI_Block" : model.SCI_Block, 
            #"K" : args.K, 
            "L" : args.L,
            "horizon" : args.horizon
            }
        # define model, loss, and optimizer
        # model = model.stackedSCI(**net_args).float().to(device)
        model = model.SCI_Net(**net_args).float().to(device)
        
    if args.model=='baseline_CNN':
         import models.baseline_model as model
         model = model.baseline_CNN(args.batch_size, args.horizon, args.seq_size).to(device)
                 
    criterion = nn.L1Loss().to(device) # L1loss is used in paper
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    # initialize loss 
    train_losses = []
    val_losses = []
 
    # run *epoch* epochs
    for epoch in range(args.epochs):
        # get timestamp for each epoch
        train_start = time.strftime("%Y%m%d-%H%M")
        # train model and save loss value
        train_loss = Train(model, train_queue, optimizer, criterion, device, args.num_steps, args.report_freq, args.horizon)
        train_losses.append(train_loss)
        print("Epoch", epoch, " Train loss: ", train_loss)
        
        # validate model and save loss value
        val_loss = Valid(model, val_queue, optimizer, criterion, device, args.val_num_steps, args.report_freq, args.horizon)   
        val_losses.append(val_loss)
        print("Epoch", epoch, " Validation loss: ", val_loss)
        
    torch.save(model, args.model_path)
    valloss_file = '{}-val_loss-{}'.format(args.save, train_start)
    np.save(valloss_file, val_losses)
    trainloss_file = '{}-train_loss-{}'.format(args.save, train_start)
    np.save(trainloss_file, train_losses)  
    
      
    
if __name__ == '__main__':
  main() 
  