# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:46:13 2019

@author: Thulitha t-T-s 
@reg No: ********
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import pandas as pd

dataset =pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
y = y.astype(float).reshape(30,1)

#Splitting into training and Testing data
###############################################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
test = np.concatenate((X_test, y_test), axis=1)

#Simple linear regression analysis related functions 
###############################################################################
def calc_coeffients(X,Y):
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)

    covariance = 0
    variance = 0
    for i in range(len(X)):
        covariance += (X[i] - mean_X) * (Y[i] - mean_Y)
        variance += (X[i] - mean_X) ** 2
        gradient = covariance / variance
        intercept = mean_Y - (gradient * mean_X)
    return [gradient, intercept];

def simple_linear_regression(train, test):
	predictions = list()
	gradient, intercept = calc_coeffients(train[0],train[1])
	for row in test:
		y_hat = intercept + gradient * row[0]
		predictions.append(y_hat)
	return predictions

def pred_error(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

gradient, intercept = calc_coeffients(X_train,y_train)

predicts = simple_linear_regression([X_train,y_train], test)
predicts = np.asarray(predicts)

error = pred_error(test[1], predicts)

print('Coefficients: gradient=%.3f, intercept=%.3f' % (gradient, intercept))   













