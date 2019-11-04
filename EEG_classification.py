#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from utils import train_test
from utils import feach_gen
from utils.train_test import fit_predict
from utils import preprocessing
from utils.load_data import load_data
import os

def find_best(X, y):
    if  len(X.shape) == 2:
        channel = 'single'
    elif len(X.shape) == 3:
        channel = 'multi'
    # pure score first
    print('No feature generation applied')
    fit_predict(X, y, selection = True)
    # SCSA 
    print('Semi-classical signal analysis')
    fit_predict(feach_gen.gen_eigen(X), y, channel)
    # average activity
    if channel == 'multi': 
        print('Averaging of the signal')
        sec = X.shape[2]/sfreq
        fit_predict(feach_gen.avactivity(X, 0, 0, sec, sfreq), y)
    # major freqs
    print('Major frequncies analysis')
    fit_predict(feach_gen.major_freqs(X), y, selection = True)
    
    # simple neural network
    # 1D CNN
    return 0
       

filepath = '/data/data.csv'
filename = os.getcwd() + filepath
data, X, y = load_data(filename, header = 0, index_col = 0, binarize = True)
# define samplig rate (datapoints per second)
sfreq = 178
# Filter the data, and plot both the original and filtered signals.
X_flt = preprocessing.butter_bandpass_filter(X, 
                                             lowcut = 0.01, 
                                             highcut = 40, 
                                             sampling_rate = 178, 
                                             order = 5, 
                                             how_to_filt = 'simultaneously')

# Try different feature generation methods for ML classifiers
find_best(X_flt, y)
