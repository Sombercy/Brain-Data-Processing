#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from utils import train_test
#from utils.feach_gen import gen_eigen
from utils.train_test import fit_predict
from utils import preprocessing
from utils.load_data import load_data
import os

filepath = '/data/data.csv'
filename = os.getcwd() + filepath
data, X, y = load_data(filename, header = 0, index_col = 0, binarize = True)
# Filter the data, and plot both the original and filtered signals.
X_flt = preprocessing.butter_bandpass_filter(X, 
                                             lowcut = 0.01, 
                                             highcut = 40, 
                                             sampling_rate = 178, 
                                             order = 5, 
                                             how_to_filt = 'simultaneously')
"""
X = gen_eigen(signal)
acc, roc_auc, feach_num = fit_predict(X, y)
print('Best accuracy: %f \nBest roc-auc: %f \nNumber of features: %s' % (acc,roc_auc,feach_num))
"""