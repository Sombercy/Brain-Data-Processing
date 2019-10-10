#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:31:37 2019

@author: savenko
"""
from utils import loo_estimators
from sklearn.preprocessing import StandardScaler


def fit_predict(X, y , full_name):
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pure_acc = loo_estimators.loo_estimator(X_scaled, y)
    # choose range of features by yourself
    delta = X_scaled.shape[1]//100
    max_score  = 0
    best_k = 0
    for k in range(delta, X_scaled.shape[1], delta):
        score = loo_estimators.ANOVA_loo_estimator(X_scaled, y, k)
        if score > max_score: 
            max_score = score
            best_k = k
    print('%f accuracy with %s features for %s' % (max_score, best_k, full_name))
    return pure_acc, max_score, best_k