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
    This function returns maximum accuracy score, maximum roc_auc score and 
    size of feature vector
    """
    # Normilize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scoring = 'roc_auc'
    max_auc = 0
    # Evaluate on different Classifiers
    score = cross_val_score(SVC(kernel='linear', gamma='auto',
                               C=1, class_weight='balanced'), 
                     X_scaled, y, cv=KFold(n_splits=10).split(X, y), scoring=scoring, n_jobs=-1)
    if np.mean(score) > max_auc: max_auc = np.mean(score)
    print('Mean SVC ROC_AUC: ', np.mean(score))
    print('SVC ROC_AUC std: ', np.std(score))
    score = cross_val_score(LogisticRegression(random_state=0, solver='liblinear'), 
                            X_scaled, y, cv=KFold(n_splits=10).split(X, y), scoring=scoring, n_jobs=-1)
    if np.mean(score) > max_auc: max_auc = np.mean(score)
    print('Mean LR ROC_AUC: ', np.mean(score))
    print('LR ROC_AUC std: ', np.std(score))
    score = cross_val_score(DecisionTreeClassifier(random_state=0), 
                            X_scaled, y, cv=KFold(n_splits=10).split(X, y), scoring=scoring, n_jobs=-1)
    if np.mean(score) > max_auc: max_auc = np.mean(score)
    print('Mean DT ROC_AUC: ', np.mean(score))
    print('DT ROC_AUC std: ', np.std(score))
    
    scoring = 'accuracy'
    max_acc = 0
    # Evaluate on different Classifiers
    score = cross_val_score(SVC(kernel='linear', gamma='auto',
                                C=1, class_weight='balanced'), 
                     X_scaled, y, cv=KFold(n_splits=10).split(X, y), scoring=scoring, n_jobs=-1)
    if np.mean(score) > max_acc: max_acc = np.mean(score)
    print('Mean SVC Accuracy: ', np.mean(score))
    print('SVC Accuracy std: ', np.std(score))
    score = cross_val_score(LogisticRegression(random_state=0, solver='liblinear'), 
                            X_scaled, y, cv=KFold(n_splits=10).split(X, y), scoring=scoring, n_jobs=-1)
    if np.mean(score) > max_acc: max_acc = np.mean(score)
    print('Mean LR Accuracy: ', np.mean(score))
    print('LR Accuracy std: ', np.std(score))
    score = cross_val_score(DecisionTreeClassifier(random_state=0), 
                            X_scaled, y, cv=KFold(n_splits=10).split(X, y), scoring=scoring, n_jobs=-1)
    if np.mean(score) > max_acc: max_acc = np.mean(score)
    print('Mean DT Accuracy: ', np.mean(score))
    print('DT Accuracy std: ', np.std(score))
    
    return max_acc, max_auc, X.shape[1]
