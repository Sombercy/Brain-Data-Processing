#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:21:40 2019

@author: savenko
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import computations as comp
import functional as func
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
    
data = pd.read_csv('data.csv')
labels = data['y'].values.tolist()
labels = [x if x == 1 else 0 for x in labels]
data = data.drop(data.columns[[0,-1]], axis = 1)
data = data - data.min().min()

data['y'] = labels
X = data[data.columns[:-1]].values

h = max(X[500])/20
yscsa, kappa, n, psinnor = func.scsa(np.array(X[500]),h)
scaler = StandardScaler()
y = scaler.fit_transform(X[500].reshape(-1, 1))[:, 0]
yscsa = scaler.fit_transform(yscsa.reshape(-1, 1))[:, 0]
t = np.linspace(0,1, 178)
fig, ax = plt.subplots(figsize=(12, 8), facecolor = 'w')
line1, = ax.plot(t, y, color = 'k', label = 'signal')
line2, = ax.plot(t, yscsa, color = 'r', label = 'SCSA reconstruction' %h)
line2.set_dashes([2, 2, 10, 2])
line3, = ax.plot(t, y-yscsa, color ='gold', label = 'difference' %h)
ax.set_title('SCSA reconstruction of the EEG signal')
plt.xlabel('Time, s')
plt.ylabel('Scaled Amplitude')
"""
h = max(X[500])/100
yscsa, kappa, n, psinnor = func.scsa(np.array(X[500]),h)
yscsa = scaler.fit_transform(yscsa.reshape(-1, 1))[:, 0]
line4, = ax.plot(t, y, color = 'palevioletred', label = 'SCSA (h=%s)' %h)
line4.set_dashes([6, 2])
line3, = ax.plot(t, y-yscsa, color ='rosybrown', label = 'signal - SCSA(h=%s)' %h)
"""
ax.legend()
plt.show()

"""
#yscsa, kappa, n, psinnor = comp.eigen(np.array(df))
eigens = []
hs = []
Nh = []
for k in range(data.shape[0]):
    temp = list(data.iloc[k])
    h = max(temp)/20
    hs.append(h)
    yscsa, kappa, n, psinnor = func.scsa(np.array(temp),h)
    Nh.append(n)
    eigens.append(kappa.diagonal().tolist())

Nh = min(Nh)
X = np.array([x[:Nh] for x in eigens])
y = np.array(labels)

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('clf', SVC(kernel='rbf', gamma='auto', 
                C=1, class_weight='balanced'))])


kf = KFold(n_splits=10).split(X, y)
scoring = 'roc_auc'
# evaluate pipeline
score = cross_val_score(pipe, X, y, cv=kf, scoring=scoring, n_jobs=-1)
print('Mean ROC_AUC: ', np.mean(score))
print('ROC_AUC std: ', np.std(score))
  
kf = KFold(n_splits=5, shuffle = True)
kf.get_n_splits(X)
accuracy = []
X = np.array(X)
y = np.array(y)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)
    accuracy.append(clf.score(X_test, y_test))

print(np.mean(accuracy))"""
