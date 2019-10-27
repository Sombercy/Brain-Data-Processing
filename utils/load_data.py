#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from scipy.io import loadmat
import pandas as pd

def load_data(filename, index = False, binary = True):
    if filename[-3:] == 'csv': 
        df = pd.read_csv(filename)
        if index: 
            df = df.drop(df.columns[0], axis=1)
        X = df.drop(['y'], axis = 1).values
        y = df['y'].values
        if not binary:
            # binarize target
            y[y>1] = 0
    elif filename[-3:] == 'mat':
        df = loadmat(filename, squeeze_me=True)
        X = df['X'].values
        y = df['y'].values
        if not binary:
            # binarize target
            y[y>1] = 0
    return X, y
