#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import model_selection
from train_test_models import train_test_models, train_test_CMC

def cross_val_loop(X, y, models, latent_spaces, betas, hidden_size, K, batch_size=100, num_epochs=20, L=5, output_file = None):
    
    n_models = len(models)
    n_spaces = len(latent_spaces)
    n_betas = len(betas)
    
    error_train = np.empty((K, n_spaces, n_betas, n_models))
    error_test = np.empty((K, n_spaces, n_betas, n_models))
    TN = np.empty((K, n_spaces, n_betas, n_models))
    FN = np.empty((K, n_spaces, n_betas, n_models))
    FP = np.empty((K, n_spaces, n_betas, n_models))
    TP = np.empty((K, n_spaces, n_betas, n_models))
    F1 = np.empty((K, n_spaces, n_betas, n_models))
    
    CV = model_selection.StratifiedKFold(K, shuffle=True, random_state=2)
    # CV = model_selection.KFold(K, shuffle=True)
    
    k=0
    for train_index, test_index in CV.split(X,y):
        
        print(f'\t Inner loop: {k+1}/{K}')
        if output_file:        
            output_file.write(f'\t Inner loop: {k+1}/{K}\n')
            output_file.flush()
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        for m, model in enumerate(models):
            if model.__name__ == 'CMC':
                (error_train[k,:,:,m], error_test[k,:,:,m], TN[k,:,:,m], 
                    FN[k,:,:,m], FP[k,:,:,m], TP[k,:,:,m], F1[k,:,:,m]) = train_test_CMC(X_train, y_train, X_test, y_test, 
                                                                                         model, hidden_size, batch_size, 
                                                                                         num_epochs, output_file = output_file)
            else:
                for s, latent_features in enumerate(latent_spaces):
                    for b, beta in enumerate(betas):
                        (error_train[k,s,b,m], error_test[k,s,b,m], TN[k,s,b,m], 
                        FN[k,s,b,m], FP[k,s,b,m], TP[k,s,b,m], F1[k,s,b,m]) = train_test_models(X_train, y_train, X_test, y_test, 
                                                                                  model, latent_features, beta, hidden_size, 
                                                                                  batch_size, num_epochs, L, output_file = output_file)
        
        k += 1
    
    # Minimize error
    # t = np.mean(error_test, axis=0)
    # opt_val_err = np.min(t, axis=(0,1))
    
    # indices = [np.argmin(t[:,:,k]) for k in range(n_models)]
    # opt_indices = [(indices[k]//n_betas, indices[k]%n_betas) for k in range(n_models)]
    # opt_values = [(latent_spaces[el[0]], betas[el[1]]) for el in opt_indices]
    
    # Maximize recall or F1
    t = np.mean(TP/(TP+FN), axis=0)
    # t = np.mean(F1, axis=0)
    opt_val_err = np.max(t, axis=(0,1))
    
    indices = [np.argmax(t[:,:,k]) for k in range(n_models)]
    opt_indices = [(indices[k]//n_betas, indices[k]%n_betas) for k in range(n_models)]
    opt_values = [(latent_spaces[el[0]], betas[el[1]]) for el in opt_indices]
    
    return opt_val_err, opt_values, error_test
