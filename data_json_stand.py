#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:39:38 2021

@author: nikolaj
"""
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(anomaly_threshold):

    GMdfr = pd.read_json('DATA.json')
    
    GMdata_r = torch.Tensor(GMdfr['GM.acc.xyz.z_resampled'])
    
    # Standardize the data to have mean 0 and std 1
    scaler = StandardScaler()
    scaler.fit(GMdata_r)
    GMdata_r = scaler.transform(GMdata_r)
    GMdata_r = torch.from_numpy(GMdata_r).float()
    
    # Make labels for the Green mobility data based on IRI, threshold set to anomaly_thresh
    # if over anomaly_thresh it is set to anomaly
    
    # Maybe we need to turn it into a double for CMC
    GMlabels = torch.Tensor(np.array(GMdfr['IRI_mean'] > anomaly_threshold, dtype=int))
    
    X = GMdata_r
    y = GMlabels
    
    return X, y