#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:30:49 2019

@author: savenko
"""
from utils import train_test
from utils import loo_estimators
from utils import frequency_filter
from utils import loo_fit_predict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scsa 
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


class EEG:
    
    def __init__(self, filename = 'data.csv', group = False):
        self.filename = filename
        self.group = group
        
    def preproc(self, df):
        df.rename(columns={'Unnamed: 0':'indx'}, inplace=True)
        # Get patient_id and time of recording 
        if self.group: df['sec'], df['patient_id'] = df['indx'].str.split('.', 1).str
        df.drop(columns=['indx'], inplace=True)
        if self.group:
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

        
    def train_set_gen(self, df):
        labels = []
        lamdas = []
        Nh = []
        if self.group:
            # Loop across trials
            for k in df['patient_id'].unique():
                temp = df[df['patient_id'] == k]
                labels.append(temp['y'].values)
                # Loop across time frames of a trial
                for i in temp['sec']:
                    # SCSA takes as input only 1D signal
                    signal = temp[temp['sec'] == i].drop(['patient_id','sec', 'y'], axis = 1).values[0]
                    # h should be tuned for each signal manually performing the best reconstruction 
                    h = max(signal)/20
                    yscsa, kappa, n, psinnor = scsa.scsa(signal, h)
                    # Collect numbers of eigenvalues for each signal
                    Nh.append(n)
                    # Collect eigenvalues for each signal
                    lamdas.append(kappa.diagonal().tolist())  
            y = list(chain.from_iterable(labels))
        else:
            # Loop across data samples
            for i in range(len(df)):
                signal = df.iloc[i].drop('y', axis = 0).values
                # h should be tuned for each signal manually performing the best reconstruction 
                h = max(signal)/20
                yscsa, kappa, n, psinnor = scsa.scsa(signal, h)
                # Collect numbers of eigenvalues for each signal
                Nh.append(n)
                # Collect eigenvalues for each signal
                lamdas.append(kappa.diagonal().tolist()) 
            y = df['y'].values
        Nh = min(Nh)
        # Cut eigenvalues vectors to create uniform feature vectors
        X = np.array([lamdas[i][:Nh] for i in range(len(lamdas))])
        return X, y
    
    def fit_predict(self, X, y, groups):
        """
        data = pd.DataFrame(data = X)
        data['y'] = y
        data['groups'] = groups
        # Balance the data
        length = data[data['y'] == 1].values.shape[0]
        temp = data[data['y'] == 0]
        temp = temp.sample(frac=1)[:length]
        temp = temp.append(data[data['y'] == 1])
        temp = temp.sample(frac=1)
        X = temp.drop(['y', 'groups'], axis = 1).values
        y = temp['y'].values
        groups = temp['groups'].values
        """
        pipe = Pipeline([
                ('scale', StandardScaler()),
                ('clf', SVC(kernel='rbf', gamma='auto',
                            C=1, class_weight='balanced'))])
        kf = GroupKFold(n_splits=10).split(X, y, groups)
        scoring = 'roc_auc'
        # Evaluate pipeline
        score = cross_val_score(pipe, X, y, cv=kf, scoring=scoring, n_jobs=-1)
        print('Mean ROC_AUC: ', np.mean(score))
        print('ROC_AUC std: ', np.std(score))
        return np.mean(score)

    def main(self):
        data = pd.read_csv(self.filename)
        df = self.preproc(data) # df = data if no preprocessing is needed
        if self.group: vals = df.drop(['patient_id','sec', 'y'], axis = 1)
        else: vals = df.drop(['y'], axis = 1)
        # Make data non-negative for SCSA 
        vals = vals - vals.min().min()
        df[vals.columns] = vals
        X, y = self.train_set_gen(df)
        feach_size = X.shape[1]
        if self.group:
            # devide all trials by groups excluding biases in training
            groups = df['patient_id'].values
            score = self.fit_predict(X, y, groups)
        else:
            score = train_test.fit_predict(X, y)
        print("Everything is done!")
        return score, feach_size
    
    
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
    
 
my_fMRI_samples = ['data-starplus-04847-v7.mat', 'data-starplus-04820-v7.mat', 
                 'data-starplus-04799-v7.mat', 'data-starplus-05675-v7.mat',
                 'data-starplus-05680-v7.mat', 'data-starplus-05710-v7.mat']
my_NIRS_samples = ['S0%s' % i for i in range(1, 9)]
my_EEG_samples = 'data.csv'
"""
my_class = EEG(my_EEG_samples, False)
score, k = my_class.main()
"""
my_class = NIRS(my_NIRS_samples)
pure_score, score, k = my_class.main()
print(pure_score, score, k)

        
        