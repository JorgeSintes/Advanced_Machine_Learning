# -*- coding: utf-8 -*-
"""
Created on Sat May  1 22:19:06 2021

@author: cleml
"""

import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(anomaly_threshold):

    GMdfr = pd.read_json('DATA.json')
    
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
