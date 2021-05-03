#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from tabulate import tabulate
from sklearn import model_selection
from data_json import load_data
from inner_loop import cross_val_loop
from train_test_models import train_test_models, train_test_CMC
from models import CMC, Model2, Model3, Model4, Model5, Model6, Arthur, Betty


##############################################################################
############################### VARIABLES ####################################
##############################################################################

anomaly_threshold = 2.0

K_outer = 2
K_inner = 2

latent_spaces = [2, 4, 8, 16, 32]

betas = [0.3, 0.5, 0.8, 1]

hidden_size = 20

batch_size = 100
num_epochs = 30
L = 5

models_selec = [1,2,3]

file_results = 'CV_results_123.csv'
file_variables = 'variables_123.csv'
file_console = 'running_123.out'

##############################################################################
############################# DATA LOADING ###################################
##############################################################################

X, y = load_data(anomaly_threshold)


##############################################################################
############################# MODEL SET-UP ###################################
##############################################################################

total_models = [None, CMC, Model2, Model3, Model4, Model5, Model6, Arthur, Betty]
str_total_models = [None, 'CMC', 'Model2', 'Model3', 'Model4', 'Model5', 'Model6', 'Arthur', 'Betty']

models = [total_models[i] for i in models_selec]
str_models = [str_total_models[i] for i in models_selec]
n_models = len(models)


##############################################################################
############################# OUTPUT FILE ####################################
##############################################################################

f = open(file_console, 'a')


##############################################################################
########################### CROSS VALIDATION #################################
##############################################################################

CV_outer = model_selection.StratifiedKFold(K_outer, shuffle=True, random_state=1)
# CV_outer = model_selection.KFold(K_outer, shuffle=True)

error_train = np.empty((K_outer, n_models))
error_test = np.empty((K_outer, n_models))
TN = np.empty((K_outer, n_models))
FN = np.empty((K_outer, n_models))
FP = np.empty((K_outer, n_models))
TP = np.empty((K_outer, n_models))
F1 = np.empty((K_outer, n_models))


opt_params = np.empty((K_outer, n_models), dtype=object)

k_outer = 0

for train_index, test_index in CV_outer.split(X, y):
    
    print(f'Outer fold: {k_outer+1}/{K_outer}')
    f.write(f'\nOuter fold: {k_outer+1}/{K_outer}\n')
    f.flush()
    
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    _, opt_params[k_outer], _ = cross_val_loop(X_train, y_train, models, latent_spaces, betas, hidden_size, K_inner, 
                                                     batch_size, num_epochs, L, output_file = f)
    
    for m, params in enumerate(opt_params[k_outer,:]):
        
        if models[m].__name__ == 'CMC':
            (error_train[k_outer, m], error_test[k_outer, m],
            TN[k_outer, m], FN[k_outer, m],
            FP[k_outer, m], TP[k_outer, m], F1[k_outer, m]) = train_test_CMC(X_train, y_train, X_test, y_test, 
                                                                                         models[m], hidden_size, batch_size, 
                                                                                         num_epochs, K=k_outer+1, output_file = f)
        
        else:                                                                                 
            opt_latent_space = params[0]
            opt_beta = params[1]
            
            (error_train[k_outer, m], error_test[k_outer, m],
            TN[k_outer, m], FN[k_outer, m],
            FP[k_outer, m], TP[k_outer, m], F1[k_outer, m]) = train_test_models(X_train, y_train, X_test, y_test, 
                                                                                models[m], opt_latent_space, opt_beta, 
                                                                                hidden_size, batch_size, num_epochs, 
                                                                                L, K=k_outer+1, output_file = f)
    k_outer += 1
    

##############################################################################
############################### OUTPUTS ######################################
##############################################################################

#Save the results on a DataFrame and print them
cv_results = {'Outer fold': list(range(K_outer))}
cv_columns = ['Outer fold']

for i, model in enumerate(str_models):
    cv_results[model+'_latent_space'] = [opt_params[k, i][0] for k in range(K_outer)]
    cv_columns.append(model+'_latent_space')
    
    cv_results[model+'_beta'] = [opt_params[k, i][1] for k in range(K_outer)]
    cv_columns.append(model+'_beta')
    
    cv_results[model+'_accuracy'] = list(1-error_test[:,i])
    cv_columns.append(model+'_accuracy')
    
    cv_results[model+'_precision'] = list(TP[:,i] / (TP[:,i] + FP[:,i]))
    cv_columns.append(model+'_precision')
    
    cv_results[model+'_recall'] = list(TP[:,i] / (TP[:,i] + FN[:,i]))
    cv_columns.append(model+'_recall')
    
    cv_results[model+'_F1'] = list(F1[:,i])
    cv_columns.append(model+'_F1')

df_cv = pd.DataFrame(cv_results, columns = cv_columns)

variables = {
    'anomaly_threshold': anomaly_threshold,
    'K_outer': K_outer,
    'K_inner': K_inner,
    'latent_spaces': latent_spaces,
    'betas': betas,
    'hidden_size': hidden_size,
    'batch_size': batch_size,
    'num_epochs': num_epochs,
    'L': L
    }

df_variables = pd.DataFrame(variables)

print(tabulate(df_cv, headers='keys', tablefmt='psql', showindex=False))
f.write(tabulate(df_cv, headers='keys', tablefmt='psql', showindex=False))

#Save it on a file
df_cv.to_csv(r'Results/' + file_results, index = False)
df_variables.to_csv(r'Results/' + file_variables)
f.close()