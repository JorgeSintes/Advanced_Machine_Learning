#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from tabulate import tabulate
from sklearn import model_selection
from data import load_data
from inner_loop import cross_val_loop
from train_test_models import train_test_models
from models import CMC, Model2, Model3, Model4, Model5, Model6, Arthur


##############################################################################
############################### VARIABLES ####################################
##############################################################################

anomaly_threshold = 2.0

K_outer = 3
K_inner = 2

latent_spaces = np.arange(5, 8)
hidden_size = 20
beta = 1
# hidden_sizes = np.arange(5, 21)

batch_size = 100
num_epochs = 20

models_selec = [2, 3]


##############################################################################
############################# DATA LOADING ###################################
##############################################################################

X, y = load_data(anomaly_threshold)


##############################################################################
############################# MODEL SET-UP ###################################
##############################################################################

total_models = [None, CMC, Model2, Model3, Model4, Model5, Model6, Arthur]
str_total_models = [None, 'CMC', 'Model2', 'Model3', 'Model4', 'Model5', 'Model6', 'Arthur']

models = [total_models[i] for i in models_selec]
str_models = [str_total_models[i] for i in models_selec]
n_models = len(models)


##############################################################################
########################### CROSS VALIDATION #################################
##############################################################################

# CV_outer = model_selection.StratifiedKFold(K_outer, shuffle=True)
CV_outer = model_selection.KFold(K_outer, shuffle=True)

error_train = np.empty((K_outer, n_models))
error_test = np.empty((K_outer, n_models))

opt_latent_space = np.empty((K_outer, n_models), dtype=int)

k_outer = 0

for train_index, test_index in CV_outer.split(X, y):
    
    print(f'Outer fold: {k_outer+1}/{K_outer}')
    
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    _, opt_latent_space[k_outer, :], _ = cross_val_loop(X_train, y_train, models, latent_spaces, hidden_size, K_inner, 
                                                     batch_size, num_epochs, beta)
    
    for m, ls in enumerate(opt_latent_space[k_outer,:]):
        error_train[k_outer, m], error_test[k_outer, m] = train_test_models(X_train, y_train, X_test, y_test, 
                                                                            models[m], ls, hidden_size, batch_size, num_epochs, beta)
    k_outer += 1
    

#Save the results on a DataFrame and print them
results = {'Outer fold': list(range(K_outer))}
columns = ['Outer fold']

for i, model in enumerate(str_models):
    results[model+'_optimal'] = list(opt_latent_space[:,i])
    columns.append(model+'_optimal')
    
    results[model+'_error'] = list(error_test[:,i])
    columns.append(model+'_error')

df = pd.DataFrame(results, columns = columns)

print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

#Save it on a file
df.to_csv(r'CV_results.csv', index = False)