#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:36:53 2019

@author: savenko
"""


from scipy.io import loadmat
import computations as comp
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import functional as func
from sklearn.svm import SVC
from itertools import chain
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from statsmodels.tsa.api import ExponentialSmoothing



def estimator(X, y):
    
    loo = LeaveOneOut()
    cv_results = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = SVC(gamma='auto', kernel = 'rbf')
        clf.fit(X_train, y_train) 
        predict = clf.predict(X_test)
        cv_results.append(accuracy_score(y_test, predict))
    print('Accuracy: %f (%f)' % (np.mean(cv_results), np.std(cv_results)))
    
    return np.mean(cv_results)

def pipe_estimator(X, y, k):
    ### Define the dimension reduction to be used.
    # Here we use a classical univariate feature selection based on F-test,
    # namely Anova. We set the number of features to be selected to 500
    feature_selection = SelectKBest(f_classif, k=k)

    # We have our classifier (SVC), our feature selection (SelectKBest), and now,
    # we can plug them together in a *pipeline* that performs the two operations
    # successively:
    clf = SVC(gamma='auto', kernel='linear', C=0.01)
    anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])
    pipe = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=k)),
    ('clf', SVC(kernel='rbf', gamma='auto', 
                C=1, class_weight='balanced'))])
    loo = LeaveOneOut()
    cv_results = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        cv_results.append(accuracy_score(y_test, y_pred))

    print('Accuracy: %f (%f)' % (np.mean(cv_results), np.std(cv_results)))

    return np.mean(cv_results)


file = loadmat('S08.mat', squeeze_me=True)
data = file['data']
ndat = [data[i]['X'].tolist() for i in range(len(data))]
trials = [data[i]['trial'].tolist() for i in range(len(data))]
ROI1 = [45,46,47]
ROI2 = [17,27, 28]
ROI3 = [12, 22, 23]


X = [[ndat[i][trials[i][j-1]:trials[i][j]] if j!= 0 else ndat[i][0:trials[i][j]] for j in range(len(trials[i]))] for i in range(len(ndat))]
XX = list(chain.from_iterable(X))
rows = min([len(XX[i]) for i in range(len(XX))])
oxy = [np.mean(XX[i][:120, :52], axis = 1) for i in range(len(XX))]
deoxy = [np.mean(XX[i][:120,52:104], axis = 1) for i in range(len(XX))]
total = [np.mean(XX[i][:120,104:156], axis = 1) for i in range(len(XX))]
X = np.array([oxy[i].tolist()+deoxy[i].tolist()+total[i].tolist() for i in range(len(total))])
Y =  list(chain.from_iterable([data[i]['y'].tolist() for i in range(len(data))]))
Y = np.array(Y)%2
"""
# FFT filtering
signal = X
W = fftfreq(signal.shape[1], d=0.1)
f_signal = rfft(signal)
cut_signal = []
for fsgn in f_signal:
    cut_f_signal = fsgn.copy()
    cut_f_signal[(W<1)] = 0
    cut_signal.append(irfft(cut_f_signal))
# ----------------------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pure_svc_score = estimator(X_scaled, Y)
#k_arr = [100, 200, 500, 1000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000, 16000, 18000]
k_arr = [10, 20, 50, 100, 120, 200, 300, 360]
for k in k_arr:
    print(k)
    score = pipe_estimator(X_scaled, Y, k)
 """   

minv = X.min()
X -= minv
lamda = []
Nh = []
kaps = []

for i in range(len(X)):

    h = max(X[i])
    yscsa, kappa, n, psinnor = comp.eigen(X[i], h)
    Nh.append(n)
    eigen = kappa.diagonal()
    lamda.append(eigen.tolist())
    kaps.append([4*h*np.sum(eigen), (16/3)*(h**2)*np.sum(eigen**3)])

    
N = min(Nh)
feach = np.array([lamda[i][:N] for i in range(len(lamda))])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(np.array(feach))
pure_svc_score = estimator(X_scaled, Y)

k_arr = [1, 5, 10, 20, 30, 40]  
for k in k_arr:
    print(k)
    score = pipe_estimator(X_scaled, Y, k)