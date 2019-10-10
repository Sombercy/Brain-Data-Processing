#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:06:02 2019

@author: savenko
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import pandas as pd


def fit_predict(X, y):
    """
    data = pd.DataFrame(data = X)
    data['y'] = y
    # Balance and shuffle the data
    length = data[data['y'] == 1].values.shape[0]
    temp = data[data['y'] == 0]
    temp = temp.sample(frac=1)[:length]
    temp = temp.append(data[data['y'] == 1])
    temp = temp.sample(frac=1)
    
    X = temp.drop('y', axis = 1).values
    y = temp['y'].values
    """
    # Normilize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scoring = 'roc_auc'
    # Evaluate on different Classifiers
    score = cross_val_score(SVC(kernel='linear', gamma='auto',
                                C=1, class_weight='balanced'), 
                     X_scaled, y, cv=KFold(n_splits=10).split(X, y), scoring=scoring, n_jobs=-1)
    print('Mean SVC ROC_AUC: ', np.mean(score))
    print('SVC ROC_AUC std: ', np.std(score))
    score = cross_val_score(LogisticRegression(random_state=0, solver='liblinear'), 
                            X_scaled, y, cv=KFold(n_splits=10).split(X, y), scoring=scoring, n_jobs=-1)
    print('Mean LR ROC_AUC: ', np.mean(score))
    print('LR ROC_AUC std: ', np.std(score))
    score = cross_val_score(DecisionTreeClassifier(random_state=0), 
                            X_scaled, y, cv=KFold(n_splits=10).split(X, y), scoring=scoring, n_jobs=-1)
    print('Mean DT ROC_AUC: ', np.mean(score))
    print('DT ROC_AUC std: ', np.std(score))
    return np.mean(score)