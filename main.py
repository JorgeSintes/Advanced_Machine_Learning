#!/usr/bin/envpython3
# -*- coding: utf-8 -*-

import numpy as np
from models import CMC, Model2, Model3, Model4, Model5, Model6, Arthur
from inner_loop import cross_val_loop
from data_json import load_data
from train_test_models import train_test_models, train_test_CMC

X, y = load_data(anomaly_threshold=2)

X_train = X[:4000]
y_train = y[:4000]
X_test = X[4000:]
y_test = y[4000:]

#%%

error_train, error_test = train_test_models(X_train, y_train, X_test, y_test, Arthur, latent_features=30, hidden_size=60, batch_size=100, num_epochs=2)

#error_train, error_test = train_test_CMC(X_train, y_train, X_test, y_test, CMC, hidden_size=40, 
#                                            batch_size=100, num_epochs=2)

print('Train error: ', error_train)
print('Test error: ', error_test)

#%%

models = [Model5]
latent_spaces = np.arange(5,8)
hidden_size = 4

K = 5

opt_error, opt_size, error_table = cross_val_loop(X, y, models, latent_spaces, hidden_size, K, batch_size=100, num_epochs=20)

print(f'Error: {opt_error}. Latent size: {opt_size}')

# %%

models = [Model2, Model3]
latent_spaces = np.arange(5,8)
hidden_size = 20

K = 5

opt_error, opt_size, error_table = cross_val_loop(X, y, models, latent_spaces, hidden_size, K, batch_size=100, num_epochs=20)