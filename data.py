#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 16:03:26 2021

@author: yorch
"""

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle5 as pickle


def load_data(anomaly_threshold):

    # Define path for getting Green Mobility data
    GMdata_r_path = 'resampled_df.pickle'
    
    objrep = open(GMdata_r_path,'rb')
    GMdfr = pickle.load(objrep)
    objrep.close()
    
    # Conversion of GMdfr to tuple of tensors (format for feeding into GRU-VAE)
    # list_tens = [torch.from_numpy(x) for x in GMdfr['GM.acc.xyz.z_resampled']]
    # GMdata_r = torch.stack(list_tens,dim=0)
    # GMdata_r = GMdata_r.float()
    
    GMdata_r = torch.Tensor(GMdfr['GM.acc.xyz.z_resampled'])
    
    # scale the z-axis acceleration data into range [a,b]
    a = -1
    b = 1
    scaler = MinMaxScaler(feature_range=(a, b))
    scaler.fit(GMdata_r)
    GMdata_r = scaler.transform(GMdata_r)
    GMdata_r = torch.from_numpy(GMdata_r)
    GMdata_r = GMdata_r.float()
    
    # Make labels for the Green mobility data based on IRI, threshold set to anomaly_thresh
    # if over anomaly_thresh it is set to anomaly
    
    # Maybe we need to turn it into a double for CMC
    GMlabels = torch.Tensor(np.array(GMdfr['IRI_mean'] > anomaly_threshold, dtype=int))
    
    X = GMdata_r
    y = GMlabels
    
    return X, y