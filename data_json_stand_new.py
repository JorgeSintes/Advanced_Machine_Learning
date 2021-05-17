#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:39:38 2021

@author: nikolaj
"""
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_data(anomaly_threshold=None):

    df = pd.read_json('DATA.json')
    
    data = torch.Tensor(df['GM.acc.xyz.z_resampled'])
    
    # Standardize the data to have mean 0 and std 1
    scaler = StandardScaler()
    #print(data.size())
    data = torch.transpose(data,0,1)
    #print(data.size())
    scaler.fit(data)
    data = scaler.transform(data)
    data = data.T
    #print(data.shape)
    
    y = torch.Tensor(np.array(df['IRI_mean'] > anomaly_threshold, dtype=int))
    
    X = torch.from_numpy(data).float()
    
    return X, y