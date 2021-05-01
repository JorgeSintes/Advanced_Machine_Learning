#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
from models import VariationalInference

def train_test_models(X_train, y_train, X_test, y_test, model, latent_features, hidden_size, batch_size=100, num_epochs=20, beta=1):
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
    # print(f">> Using device: {device}. Training {model}")
        
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
        
        r = (torch.sum(y_train)/len(y_train)).item()
        
        px_threshold = np.percentile(px_train, 100*r)
        
        y_pred_train = np.array(px_train < px_threshold, dtype=float)
        y_pred_test = np.array(px_test < px_threshold, dtype=float)
        
        error_train = np.abs((y_train - y_pred_train)).mean()
        error_test = np.abs((y_test - y_pred_test)).mean()
    
    return error_train.item(), error_test.item()


def train_test_CMC(X_train, y_train, X_test, y_test, cmc, hidden_size, batch_size=100, num_epochs=20):
    
    input_shape = X_train[0].shape
    sequence_length = X_train.size(1)
    num_layers = 1
    learning_rate = 1e-3
    
    X_train_batches = torch.split(X_train, batch_size, dim=0)
    y_train_batches = torch.split(y_train, batch_size, dim=0)
    # X_test_batches = torch.split(X_test, batch_size, dim=0)
    # y_test_batches = torch.split(y_test, batch_size, dim=0)
    
    model = cmc(input_shape, hidden_size, num_layers)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    
    epoch = 0    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f">> Using device: {device}. Training {model}")
        
    # move the model to the device
    model = model.to(device)
        
    # training..
    while epoch < num_epochs:
        epoch += 1
        print(epoch)
        # model.train()
        
        for x, y in zip(X_train_batches, y_train_batches):
            x = x.to(device)
            y = y.to(device)
            
            # perform a forward pass through the model and compute the ELBO
            y_pred = model(x)
            loss = criterion(y_pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    with torch.no_grad():
        model.eval()
        
         # Load all the training and test data without batches
        x_train = X_train
        x_test = X_test

        x_train = x_train.to(device)
        x_test = x_test.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        y_prob_train = torch.Tensor.cpu(model(x_train))
        y_prob_test = torch.Tensor.cpu(model(x_test))
        
        y_pred_train = np.array(y_prob_train >= 0.5, dtype=float)
        y_pred_test = np.array(y_prob_test >= 0.5, dtype=float)
        
        error_train = np.abs((y_train - y_pred_train)).mean()
        error_test = np.abs((y_test - y_pred_test)).mean()
    
    return error_train.item(), error_test.item()
