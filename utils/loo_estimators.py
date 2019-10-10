#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:55:50 2019

@author: savenko
"""
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif


def loo_estimator(X, y):
    
    loo = LeaveOneOut()
    cv_results = []
    lr_results = []
    dt_results = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = SVC(gamma='auto', kernel = 'linear',
                C=1, class_weight='balanced')
        clf.fit(X_train, y_train) 
        predict = clf.predict(X_test)
        cv_results.append(accuracy_score(y_test, predict))
        clf = LogisticRegression(random_state=0, solver='liblinear')
        clf.fit(X_train, y_train) 
        predict = clf.predict(X_test)
        lr_results.append(accuracy_score(y_test, predict))
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X_train, y_train) 
        predict = clf.predict(X_test)
        dt_results.append(accuracy_score(y_test, predict))
    print('SVC Accuracy: %f (%f)' % (np.mean(cv_results), np.std(cv_results)))
    print('LR Accuracy: %f (%f)' % (np.mean(lr_results), np.std(lr_results)))
    print('DT Accuracy: %f (%f)' % (np.mean(dt_results), np.std(dt_results)))
    best_score = max([np.mean(cv_results), np.mean(lr_results), np.mean(dt_results)])
    return best_score

def ANOVA_loo_estimator(X, y, k):
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

    return np.mean(cv_results)