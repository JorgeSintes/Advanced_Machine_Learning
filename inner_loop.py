#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import model_selection
from train_test_models import train_test_models
from models import CMC

def cross_val_loop(X, y, models, latent_spaces, hidden_size, K, batch_size=100, num_epochs=20, beta=1):
    
    n_models = len(models)
    n_spaces = len(latent_spaces)
    
    error_train = np.empty((K, n_spaces, n_models))
    error_test = np.empty((K, n_spaces, n_models))
    
    # CV = model_selection.StratifiedKFold(K, shuffle=True)
    CV = model_selection.KFold(K, shuffle=True)
    
    k=0
    for train_index, test_index in CV.split(X,y):
        
        print(f'\t Inner loop: {k+1}/{K}')
        
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        for s, latent_features in enumerate(latent_spaces):
                for m, model in enumerate(models):
                    error_train[k,s,m], error_test[k,s,m] = train_test_models(X_train, y_train, X_test, y_test, 
                                                                              model, latent_features, hidden_size, 
                                                                              batch_size, num_epochs, beta)
        
        k += 1
    
    error_table = error_test   
    opt_val_err = np.min(np.mean(error_test,axis=0), axis=0)
    opt_latent_size = [latent_spaces[k] for k in np.argmin(np.mean(error_test, axis=0), axis=0)]
        
    return opt_val_err, opt_latent_size, error_table    
