# -*- coding: utf-8 -*-
"""
Created on Wed May  5 21:45:34 2021

@author: cleml
"""

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def load_data(anomaly_threshold=None):

    df = pd.read_json('Sampled_data.json')
    
    data = torch.Tensor(df['data'])
    
    # Standardize the data to have mean 0 and std 1
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    
    X = torch.from_numpy(data).float()
    y = data = torch.Tensor(df['labels'])
    
    return X, y