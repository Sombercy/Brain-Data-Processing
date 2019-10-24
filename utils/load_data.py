#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:28:10 2019

@author: savenko
"""
from scipy.io import loadmat
import pandas as pd

def load_data(filename, ftype = 'csv'):
    if ftype == 'csv': 
        df = pd.read_csv(filename)
        X = df.drop(['y'], axis = 1)
        y = df['y']
    elif ftype == 'mat':
        df = loadmat(full_name, squeeze_me=True)
        X = df['X']
        y = df['y']
    return X, y
 
        

        
    