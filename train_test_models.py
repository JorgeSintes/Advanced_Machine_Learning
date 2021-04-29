#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from models import VariationalInference

def train_test_models(X_train, y_train, X_test, y_test, model, latent_features, hidden_size, batch_size=100, num_epochs=20):
    '''
    Train and test a model in particular
    '''
    
    input_shape = X_train[0].shape
    sequence_length = X_train.size(1)
    num_layers = 1
    learning_rate = 1e-3
    
    X_train_batches = torch.split(X_train, batch_size, dim=0)
    # y_train_batches = torch.split(y_train, batch_size, dim=0)
    # X_test_batches = torch.split(X_test, batch_size, dim=0)
    # y_test_batches = torch.split(y_test, batch_size, dim=0)
    
    # VAE
    vae = model(input_shape, latent_features, hidden_size, sequence_length, num_layers)
    
    # Evaluator: Variational Inference
    beta = 1
    vi = VariationalInference(beta=beta)
    
    # The Adam optimizer works really well with VAEs.
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    
    epoch = 0    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Using device: {device}")
    
    # move the model to the device
    vae = vae.to(device)
    
    # training..
    while epoch < num_epochs:
        epoch += 1
        vae.train()
        
        # Go through each batch in the training dataset using the loader
        # Note that y is not necessarily known as it is here
        for x in X_train_batches:
            x = x.to(device)
            
            # perform a forward pass through the model and compute the ELBO
            loss, diagnostics, outputs = vi(vae, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Calculating train and test error
    with torch.no_grad():
        vae.eval()
        
        # Load all the training and test data without batches
        x_train = X_train
        x_test = X_test

        x_train = x_train.to(device)
        x_test = x_test.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        loss_train, diagnostics_train, outputs_train = vi(vae, x_train)
        loss_test, diagnostics_test, outputs_test = vi(vae, x_test)
        
        px_train = torch.Tensor.cpu(diagnostics_train['log_px'])
        px_test = torch.Tensor.cpu(diagnostics_test['log_px'])
        
        r = (torch.sum(y_train) / len(y_train)).item()
        
        px_threshold = np.percentile(px_train, r)
        
        y_pred_train = np.array(px_train < px_threshold, dtype=float)
        y_pred_test = np.array(px_test < px_threshold, dtype=float)
        
        error_train = np.abs((y_train - y_pred_train)).mean()
        error_test = np.abs((y_test - y_pred_test)).mean()
    
    return error_train.item(), error_test.item()