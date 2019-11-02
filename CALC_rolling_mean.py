#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:44:02 2019

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
from scipy.fftpack import fft
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from scipy.signal import savgol_filter
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing


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

    loo = LeaveOneOut()
    cv_results = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        anova_svc.fit(X_train, y_train)
        y_pred = anova_svc.predict(X_test)
        cv_results.append(accuracy_score(y_test, y_pred))

    print('Accuracy: %f (%f)' % (np.mean(cv_results), np.std(cv_results)))

    return np.mean(cv_results)


files = ['data-starplus-04847-v7.mat', 'data-starplus-04820-v7.mat', 
         'data-starplus-04799-v7.mat', 'data-starplus-05675-v7.mat',
         'data-starplus-05680-v7.mat', 'data-starplus-05710-v7.mat']
cols = ["CALC", "LIPL", "LT", "LTRIA", "LOPER", "LIPS", "LDLPFC"]
y = []
signal = []
minimum = 0
for fname in files:
    fname = 'data-starplus-04847-v7.mat'
    file = loadmat(fname, squeeze_me=True)
    colToROI = file['meta']['colToROI'].tolist()
    cond = file['info']['cond']
    trials = [i for i in range(len(cond)) if cond[i] in (2, 3)]
#calc = [i for i in range(len(colToROI)) if colToROI[i] in cols]
    glob = file['data'][trials].tolist()
    
    data = [glob[i][:, [j for j in range(len(colToROI)) if colToROI[j] == cols[0]]] for i in range(len(glob))]


    pic = []
    sen = []
    stimulus = file['info']['firstStimulus'][trials].tolist()
    for i in range(len(data)):
        temp = data[i]
        #if stimulus[i] == 'P': pic.append(np.array(temp[:8, :]))
        #else: sen.append(np.array(temp[:8, :]))
        pcr = [temp[:16, :] if stimulus[i] == 'P' else temp[16:32, :]]
        sns = [temp[:16, :] if stimulus[i] == 'S' else temp[16:32, :]]
        pcr = np.array(pcr[0])
        sns = np.array(sns[0])
        pcr = [pcr[j] - np.mean(glob[i][j]) if stimulus[i] == 'P' else pcr[j] - np.mean(glob[i][16+j]) for j in range(pcr.shape[0])]
        sns = [sns[j] - np.mean(glob[i][j]) if stimulus[i] == 'S' else sns[j] - np.mean(glob[i][16+j]) for j in range(sns.shape[0])]
        
        
        pic.append(np.array(pcr))
        sen.append(np.array(sns))


    y += [1]*len(pic) + [0]*len(sen)
    dtemp = pic + sen
    dtemp = [list(chain.from_iterable(dtemp[i].T)) for i in range(len(dtemp))]
    if np.array(dtemp).min() < minimum: minimum = np.array(dtemp).min()
    signal += dtemp
    break

y = np.array(y)
"""pure_svc_score = estimator(np.array(signal), y)
k_arr = [20, 50, 100, 200, 300, 400, 500]
for k in k_arr:
    print(k)
    score = pipe_estimator(np.array(signal), y, k)
"""

#X =  pd.DataFrame([np.array(signal[i]) - minimum for i in range(len(signal))])
X = pd.DataFrame(signal)
#X_rol = X.rolling(15, axis = 1).mean().drop([x for x in range(15)], axis = 1)
rol1 = X.rolling(15, axis = 1).mean().drop([x for x in range(15)], axis = 1).values
rol2 = X.rolling(75, axis = 1).mean().drop([x for x in range(75)], axis = 1).values
"""
model = SimpleExpSmoothing(X.iloc[12][:300].values.copy(order = 'C'))

fit1 = model.fit()
fit2 = model.fit(smoothing_level=.2)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(X.iloc[12][:300], color = 'slategray')
ax.plot(rol1[12][:300], color = 'coral', label = 'width=15')
ax.plot(rol2[12][:300], color = 'plum', label = 'width=35')
for f, c in zip((fit1, fit2),('lime','c')):
    ax.plot(f.fittedvalues, label="alpha="+str(f.params['smoothing_level'])[:3], color=c)
plt.title("Simple Smoothing")    
plt.legend()
"""
"""
lamda = []
kaps = []
Nh = []
for i in range(len(rol1)):
    h = rol1[i].max()*20
    yscsa, kappa, n, psinnor = comp.eigen(rol1[i], h)
    Nh.append(n)
    eigen = kappa.diagonal()
    lamda.append(eigen.tolist())
    kaps.append([4*h*np.sum(eigen), (16/3)*(h**2)*np.sum(eigen**3)])
    print(i)

N = min(Nh)
feach = np.array([lamda[i][:N] for i in range(len(lamda))])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(np.array(feach))
pure_scsa_score = estimator(X_scaled, y)
"""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_arr = [10, 20, 50, 100]
for k in k_arr:
    print(k)
    score = pipe_estimator(X_scaled, y, k)
    
delta = X_scaled.shape[1]//100
max_score  = 0
best_k = 0
for k in range(delta, X_scaled.shape[1], delta):
    score = pipe_estimator(X_scaled, y, k)
    if score > max_score: 
        max_score = score
        best_k = k
print('%f accuracy with %s features for %s' % (max_score, best_k, fname))

