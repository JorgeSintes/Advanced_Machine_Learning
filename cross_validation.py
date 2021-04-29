#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from sklearn import model_selection
from data import load_data
from inner_loop import cross_val_loop
from train_test_models import train_test_models
from models import Model1, Model2, Model3, Model4, Model5, Model6


##############################################################################
############################### VARIABLES ####################################
##############################################################################

anomaly_threshold = 2.0

K_outer = 5
K_inner = 5

latent_spaces = np.arange(5, 16)
hidden_size = 20
# hidden_sizes = np.arange(5, 21)

models = [Model1, Model2, Model3, Model4, Model5, Model6]
n_models = len(models)


##############################################################################
############################# DATA LOADING ###################################
##############################################################################

X, y = load_data(anomaly_threshold)


##############################################################################
########################### CROSS VALIDATION #################################
##############################################################################


CV_outer = model_selection.StratifiedKFold(K_outer, shuffle=True)

error_train = np.empty((K_outer, n_models))
error_test = np.empty((K_outer, n_models))

opt_latent_space = np.empty((K_outer, n_models))

k_outer = 0

for train_index, test_index in CV_outer.split(X, y):
    
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    _, opt_latent_space[k_outer] = cross_val_loop(X_train, y_train, models, latent_spaces, hidden_size, K_inner)
    
    for m, ls in enumerate(opt_latent_space[k_outer]):
        error_train[k_outer, m], error_test[k_outer, m] = train_test_models(X_train, y_train, X_test, y_test, 
                                                                            models[m], ls, hidden_size)
    k_outer += 1
    

#Save the results on a DataFrame and print them
data = {'Outer fold': list(range(K_outer)),
'lambda': list(logistic_opt_lambda),
'Log error': list(logistic_opt_error*100),
'K': list(KNN_opt_neighbors),
'KNN error': list(KNN_opt_error*100),
'Baseline error': list(baseline_opt*100)}

df = pd.DataFrame(data, columns = ['Outer fold', 'lambda', 'Log error', 'K', 
                                   'KNN error', 'Baseline error'])
print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

#Save it on a file
df.to_csv(r'Table2_'+ str(n)+'.csv', index = False)