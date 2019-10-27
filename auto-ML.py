#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:30:49 2019

@author: savenko
"""
from utils import train_test
from utils import loo_estimators
from utils import loo_fit_predict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from itertools import chain
from scipy import stats
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.datasets.base import Bunch
from scipy.io import loadmat

from utils.feach_gen import gen_eigen
from utils.train_test import fit_predict
from utils import preprocessing
from utils.load_data import load_data



class EEG:
    
    def __init__(self, filename = 'data.csv', header = False, index = False, binary = True, nn = False):
        # define path to file with data
        self.filename = filename
        # define if there is a header in your data
        self.header = header
        # define if there is an index column in your data
        self.index = index
        # define if your dataset contain more than 2 classes
        self.binary = binary
        # define if you want use neural network as predictive model
        self.nn = nn
        
    def main(self):
        signal, y = load_data(self.filename, self.index, self.binary)
        X = gen_eigen(signal)
        acc, roc_auc, feach_num = fit_predict(X, y)
        print('Best accuracy: %f \nBest roc-auc: %f \nNumber of features: %s' % (acc,roc_auc,feach_num))
        return 0
    
class fMRI:
    
    def __init__(self, filename):
        self.filename = filename
        
    def load_data(self):
        all_subjects = []
        for full_name in (self.filename):
            print("Fetching file : %s" % full_name)
            file = loadmat(full_name, squeeze_me=True)
            colToROI = file['meta']['colToROI'].tolist()
            cond = file['info']['cond']
            trials = [i for i in range(len(cond)) if cond[i] in (2, 3)]
            cols = ["CALC"]
            data = file['data'][trials].tolist()
            df = pd.DataFrame()
            for k in cols:
                temp = [data[i][:, [j for j in range(len(colToROI))
                        if colToROI[j] == k]] for i in range(len(data))]
                df[k] = temp

            pic = pd.DataFrame(columns = cols)
            sen = pd.DataFrame(columns = cols)
            stimulus = file['info']['firstStimulus'][trials].tolist()
            for i in df.index:
                temp = df.iloc[i]
                pcr = [temp.T.iloc[j][:16] if stimulus[i] == 'P' else temp.T.iloc[j][16:32] for j in range(len(temp)) ]
                sns = [temp.T.iloc[j][:16] if stimulus[i] == 'S' else temp.T.iloc[j][16:32] for j in range(len(temp)) ]
                pcr = pd.DataFrame(data = [pcr], columns = cols, index = [i])
                pcr['mean'] = [np.mean(data[i][:16, :]) if stimulus[i] == 'P' else np.mean(data[i][16:32, :])]
                sns = pd.DataFrame(data = [sns], columns = cols, index = [i])
                sns['mean'] = [np.mean(data[i][:16, :]) if stimulus[i] == 'S' else np.mean(data[i][16:32, :])]
                pic = pic.append(pcr)
                sen = sen.append(sns)
    
            labels = [1]*pic.shape[0] + [0]*sen.shape[0]
            # "global" signal extraction
            mean = pic['mean'].values.tolist() + sen['mean'].values.tolist()
            signal = [[pic.loc[i][j] for j in cols] for i in pic.index]
            signal += [[sen.loc[i][j] for j in cols] for i in sen.index]
            signal = [[np.mean(signal[i][j], axis = 0) - mean[i]  for j in range(len(signal[i]))] for i in range(len(signal))]

            X = np.array([list(chain.from_iterable(signal[i]))for i in range(len(signal))])
            y = np.array(labels)
            # Normalize the data
            
            print("...done.")
            all_subjects.append(Bunch(data=X, target=y))
        return all_subjects
    
    def main(self):
        file = self.load_data()
        accuracy = []
        pure_acc = []
        feach_size = []
        for i, full_name in enumerate(self.filename):
            data = file[i]
            X = data.data
            y = data.target
            pacc, acc, fsize = loo_fit_predict.fit_predict(X, y, full_name)
            pure_acc.append(pacc)
            accuracy.append(acc)
            feach_size.append(fsize)
        return np.mean(pure_acc), np.mean(accuracy), np.mean(feach_size)
    
class NIRS:
    
    def __init__(self, filename):
        self.filename = filename
        
    def load_data(self):
        all_subjects = []
        for full_name in (self.filename):
            file = loadmat(full_name, squeeze_me=True)
            data = file['data']
            ndat = [data[i]['X'].tolist() for i in range(len(data))]
            trials = [data[i]['trial'].tolist() for i in range(len(data))]
            X = [[ndat[i][trials[i][j-1]:trials[i][j]] if j!= 0 else ndat[i][0:trials[i][j]] for j in range(len(trials[i]))] for i in range(len(ndat))]
            temp = list(chain.from_iterable(X))
            # rows = min([len(temp[i]) for i in range(len(temp))])
            X = [temp[i][:120] for i in range(len(temp))]
            X = np.array([list(chain.from_iterable(X[i].T)) for i in range(len(X))])
            Y =  list(chain.from_iterable([data[i]['y'].tolist() for i in range(len(data))]))
            Y = np.array(Y)%2
            all_subjects.append(Bunch(data=X, target=Y))
        return all_subjects
    
    def main(self):
        file = self.load_data()
        accuracy = []
        pure_acc = []
        feach_size = []
        for i, full_name in enumerate(self.filename):
            data = file[i]
            X = data.data
            y = data.target
            # Frequency filtering for drifts removal
            X = frequency_filter.filtering(X, fmin = 0.01)
            puracc, acc, fsize = loo_fit_predict.fit_predict(X, y, full_name)
            pure_acc.append(puracc)
            accuracy.append(acc)
            feach_size.append(fsize)
        return np.mean(pure_acc), np.mean(accuracy), np.mean(feach_size) 
    
"""
my_fMRI_samples = ['data-starplus-04847-v7.mat', 'data-starplus-04820-v7.mat', 
                 'data-starplus-04799-v7.mat', 'data-starplus-05675-v7.mat',
                 'data-starplus-05680-v7.mat', 'data-starplus-05710-v7.mat']
my_NIRS_samples = ['S0%s' % i for i in range(1, 9)]
my_EEG_samples = 'data.csv'

my_class = EEG(my_EEG_samples, False)
score, k = my_class.main()

my_class = NIRS(my_NIRS_samples)
pure_score, score, k = my_class.main()
print(pure_score, score, k)
"""
# use code below to process EEG epileptic seizure detection        
my_EEG = EEG(filename = 'data.csv', index = True, binary = False)
my_EEG.main()
