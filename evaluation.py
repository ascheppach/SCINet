# -*- coding: utf-8 -*-
"""
@author: Scheppach Amadeu, Szabo Viktoria, To Xiao-Yin
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


# Plot train and validation loss for traffic
datatrain = np.load('results/EXPsearch-try-20210825-220037-train_loss-20210826-0004_traffic.npy', allow_pickle=1)
dataval = np.load('results/EXPsearch-try-20210825-220037-val_loss-20210826-0004_traffic.npy', allow_pickle=1)

fig1=plt.figure()
plt.plot(datatrain,label='train_loss_electricity')
plt.plot(dataval,label='val_loss_electricity')
plt.legend()
fig1.savefig('results/train_val-loss_traffic.png')
plt.close(fig1)

# Plot train and validation loss for electricity
datatrain = np.load('results/EXPsearch-try-20210831-210546-train_loss-20210831-2155_electricity.npy', allow_pickle=1)
dataval = np.load('results/EXPsearch-try-20210831-210546-val_loss-20210831-2155_electricity.npy', allow_pickle=1)

fig2=plt.figure()
plt.plot(datatrain,label='train_loss_electricity')
plt.plot(dataval,label='val_loss_electricity')
plt.legend()
fig2.savefig('results/train_val-loss_electricity.png')
plt.close(fig2)
        
        
        
        
        
        
        
