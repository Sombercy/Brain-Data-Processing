#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.sparse import diags
from scipy.special import gamma
from scipy.integrate import simps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from itertools import chain
from scipy import stats
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


def delta(n, fex, feh):
    ex = np.kron([x for x in range(n-1, 0, -1)], np.ones((n,1)))
    if (n%2) == 0:
        dx = -np.pi**2/(3*feh**2)-(1/6)*np.ones((n,1))
        test_bx = -(-1)**ex*(0.5)/(np.sin(ex*feh*0.5)**2)
        test_tx =  -(-1)**(-ex)*(0.5)/(np.sin((-ex)*feh*0.5)**2)
    else:
        dx = -np.pi**2/(3*feh**2)-(1/12)*np.ones((n,1))
        test_bx = -0.5*((-1)**ex)*np.tan(ex*feh*0.5)**-1/(np.sin(ex*feh*0.5))
        test_tx = -0.5*((-1)**(-ex))*np.tan((-ex)*feh*0.5)**-1/(np.sin((-ex)*feh*0.5))

    rng = [x for x in range(-n+1, 1, 1)] + [y for y in range(n-1, 0, -1)]
    Ex = diags(np.concatenate((test_bx, dx, test_tx), axis = 1).T, np.array(rng), shape = (n, n)).toarray()
    Dx=(feh/fex)**2*Ex
    return Dx

def scsa(y, h):
    M = len(y.tolist())
    fe = 1
    feh = 2*np.pi/M
    D = delta(M,fe,feh)
    Y = np.diag(y)
    gm = 0.5
    Lcl = (1/(2*(np.pi)**0.5))*(gamma(gm+1)/gamma(gm+(3/2)))
    SC = -h*h*D-Y
    lamda, psi = np.linalg.eigh(SC)
    temp = np.diag(lamda)
    ind = np.where(temp < 0)
    temp = temp[temp < 0]
    kappa = np.diag((-temp)**gm)
    Nh = kappa.shape[0]
    psin = psi[:, ind[0]]
    I = simps(psin**2, dx = fe, axis = 0)
    psinnor = psin/I**0.5
    yscsa =((h/Lcl)*np.sum((psinnor**2)@kappa,1))**(2/(1+2*gm))
    if y.shape != yscsa.shape: yscsa = yscsa.T
    return yscsa, kappa, Nh, psinnor


def preproc(df):
    df.rename(columns={'Unnamed: 0':'indx'}, inplace=True)
    # Get patient_id and time of recording
    df['sec'], df['patient_id'] = df['indx'].str.split('.', 1).str
    df.drop(columns=['indx'], inplace=True)
    # Replace id with the corresponding numbers
    patient_map = dict(zip(df['patient_id'].unique(), list(range(1, 501))))
    df = df.replace({'patient_id': patient_map})
    df['sec'] = df['sec'].str[1:]
    df['sec'] = df['sec'].astype('int16')
    # reorder columns
    cols = ['patient_id', 'sec'] + list(df.columns[:-2])
    df = df[cols]
    # sort data by patient_id and time
    df.sort_values(['patient_id', 'sec'], inplace=True)
    # binarize target
    df['y'].loc[df['y'] > 1] = 0
    return df


data = pd.read_csv('data.csv')
df = preproc(data)
vals = df.drop(['patient_id','sec', 'y'], axis = 1)
vals = vals - vals.min().min()
df[vals.columns] = vals

ids = []
labels = []
eigens = []
Nh = []
for k in df['patient_id'].unique():
    temp = df[df['patient_id'] == k]
    labels.append(temp['y'].values)
    ids.append(temp['patient_id'].values)
    for i in temp['sec']:
        y = temp[temp['sec'] == i].drop(['patient_id','sec', 'y'], axis = 1).values[0]
        h = max(y)/20
        yscsa, kappa, n, psinnor = scsa(y, h)
        Nh.append(n)
        eigens.append(kappa.diagonal().tolist())   


N = min(Nh)
X = np.array([eigens[i][:N] for i in range(len(eigens))])
y = np.array(list(chain.from_iterable(labels)))
groups = np.array(list(chain.from_iterable(ids)))


pipe = Pipeline([
    ('scale', StandardScaler()),
    ('clf', SVC(kernel='rbf', gamma='auto', 
                C=1, class_weight='balanced'))])


gkf = GroupKFold(n_splits=10).split(X, y, groups)
scoring = 'roc_auc'
# evaluate pipeline
score = cross_val_score(pipe, X, y, cv=gkf, scoring=scoring, n_jobs=-1)
print('Mean ROC_AUC: ', np.mean(score))
print('ROC_AUC std: ', np.std(score))
