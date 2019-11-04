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
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

def fit_predict(X, y, selection = None, k = None):
    """
    This function returns maximum accuracy score, maximum roc_auc score and 
    size of feature vector
    """
    # Normilize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scoring = 'roc_auc'
    max_auc = 0
    svc = SVC(kernel='linear', gamma='auto', C=1, class_weight='balanced')
    if selection:
    # Define the dimension reduction to be used.
    # Here we use a classical univariate feature selection based on F-test,
    # namely Anova. We set the number of features to be selected to 500
    
        delta = X_scaled.shape[1]//20
        max_score  = 0
        best_k = 0
        for k in range(delta, X.shape[1], delta):
            feature_selection = SelectKBest(f_classif, k=k)
            anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])
            score = cross_val_score(anova_svc, X_scaled, y, cv=KFold(n_splits=10).split(X, y), scoring=scoring, n_jobs=-1)
            if np.mean(score) > max_score: 
                max_score = np.mean(score)
                best_k = k
            if np.mean(score) > max_auc: max_auc = np.mean(max_score)
        print('Mean ANOVA_SVC ROC_AUC: ', max_score, ' with %s features' % best_k )
    # Evaluate on different Classifiers
    score = cross_val_score(svc, X_scaled, y, cv=KFold(n_splits=10).split(X, y),
                            scoring=scoring, n_jobs=-1)
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
    if selection:
    # Define the dimension reduction to be used.
    # Here we use a classical univariate feature selection based on F-test,
    # namely Anova. We set the number of features to be selected to 500
        feature_selection = SelectKBest(f_classif, k=k)
        anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])

        delta = X_scaled.shape[1]//20
        max_score  = 0
        best_k = 0
        for k in range(delta, X.shape[1], delta):
            score = cross_val_score(anova_svc, X_scaled, y, cv=KFold(n_splits=10).split(X, y), scoring=scoring, n_jobs=-1)
            if np.mean(score) > max_score: 
                max_score = np.mean(score)
                best_k = k
            if np.mean(score) > max_auc: max_auc = np.mean(max_score)
        print('Mean ANOVA_SVC Accuracy: ', max_score, ' with %s features' % best_k )
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

    return 0
