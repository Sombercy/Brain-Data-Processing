#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:05:00 2019
karoch samoi optimal'noi okazalas' eta setka pri 16 maps
@author: savenko
"""

import scipy.io as io
import numpy as np
import math
import os
import tarfile

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy import io
from sklearn.model_selection import KFold
from sklearn.datasets.base import Bunch

#import nibabel as ni
from scipy.fftpack import fft, ifft

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_classif

import computations as comp
import functional as func

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from scipy.fftpack import fft

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, Conv3D, MaxPool3D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler



full_names = ['data-starplus-04847-v7.mat', 'data-starplus-04820-v7.mat', 
         'data-starplus-04799-v7.mat', 'data-starplus-05675-v7.mat',
         'data-starplus-05680-v7.mat', 'data-starplus-05710-v7.mat']
cols = ["CALC", "LIPL", "LT", "LTRIA", "LOPER", "LIPS", "LDLPFC"]

success_indices = []
for indice, full_name in enumerate(full_names):
    # ne sprashuvai pochemy tak ubogo sdelana iteratcuya po subject-am  >_<
    fname = 'data-starplus-04847-v7.mat'
    print("Fetching file : %s" % fname)
    # General information
    data = io.loadmat(fname)
    n_voxels = data['meta']['nvoxels'].flat[0].squeeze()
    n_trials = data['meta']['ntrials'].flat[0].squeeze()
    dim_x = data['meta']['dimx'].flat[0].squeeze()
    dim_y = data['meta']['dimy'].flat[0].squeeze()
    dim_z = data['meta']['dimz'].flat[0].squeeze()

    # Loading X
    X_temp = data['data'][:, 0]

    # Loading y
    y = data['info']
    y = y[0, :]
    # y = np.array([y[i].flat[0]['actionRT'].flat[0]
    y = np.array([y[i].flat[0]['cond'].flat[0] for i in range(n_trials)])
    good_trials = np.where(y > 1)[0]
    n_good_trials = len(good_trials)
    n_times = 16  # 8 seconds
    # sentences
    XS = np.zeros((n_good_trials, dim_x, dim_y, dim_z))
    # pictures
    XP = np.zeros((n_good_trials, dim_x, dim_y, dim_z))
    first_stim = data['info']['firstStimulus']
    
    ROI = np.zeros((dim_x, dim_y, dim_z)).astype('str')
    for j in range(n_voxels):
        x, y, z = data['meta']['colToCoord'].flat[0][j, :] - 1
        try: 
            ROI[x, y, z] = data['meta']['colToROI'].flat[0].tolist()[j][0][0]
        except ValueError:
            print('Voxel %s does not relay to any ROI' % j)
            ROI[x, y, z] = '0.0'
        
    # Averaging on the time
    for k, i_trial in enumerate(good_trials):
        i_first_stim = str(first_stim.flat[i_trial][0])
        XSk = XS[k]
        XPk = XP[k]

        for j in range(n_voxels):
            # Getting the right coords of the voxels
            x, y, z = data['meta']['colToCoord'].flat[0][j, :] - 1
            Xkxyz = X_temp[i_trial][:, j]
            # Xkxyz -= Xkxyz.mean()  # remove drifts
            if i_first_stim == 'S':  # sentence
                XSk[x, y, z] = Xkxyz[:n_times].mean()
                XPk[x, y, z] = Xkxyz[n_times:2 * n_times].mean()
            elif i_first_stim == 'P':  # picture
                XPk[x, y, z] = Xkxyz[:n_times].mean()
                XSk[x, y, z] = Xkxyz[n_times:2 * n_times].mean()
            else:
                raise ValueError('Uknown first_stim : %s' % first_stim)

    X = np.r_[XP, XS]
    y = np.ones(2 * n_good_trials)
    y[:n_good_trials] = 0

    X = X.astype(np.float)
    y = y.astype(np.float)
    break

# Plot scans we have
scaler = StandardScaler()
fig = plt.figure()
for num in range(X[1].shape[2]):
    scan = fig.add_subplot(2,4,num+1)
    scan.imshow(scaler.fit_transform(X[0][:,:,num]))
plt.show()


X_train = X.reshape(-1,64,64, 8, 1)
Y_train = to_categorical(y, num_classes = 2)

# GLOBAL VARIABLES
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)

#EXPERIMENT 2

model = Sequential()
model.add(Conv3D(16,kernel_size=(4, 4, 2),activation='relu',input_shape=(64,64,8, 1)))
model.add(Conv3D(16,kernel_size=(4, 4, 2),activation='relu', padding = 'same'))
model.add(MaxPool3D(pool_size=(2, 2, 1)))
model.add(Conv3D(32,kernel_size=(4, 4,2),activation='relu'))
model.add(Conv3D(32,kernel_size=(4, 4, 2),activation='relu'))
model.add(MaxPool3D(pool_size=(2, 2, 1)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])    
model.summary()
#LEAVE ONE OUT CROSS-VALIDATION 
loo = LeaveOneOut()
name = "16 maps"
cv_result = []

for train_index, test_index in loo.split(X_train):
    clf = model
    X_train2, X_val2 = X_train[train_index], X_train[test_index]
    Y_train2, Y_val2 = Y_train[train_index], Y_train[test_index]
    acc = clf.fit(X_train2,Y_train2, batch_size=None, epochs=1, verbose=0,
                  validation_data=(X_val2, Y_val2), workers=1, callbacks=[annealer])
    cv_result.append(acc.history['val_accuracy'])
history= np.mean(cv_result)
print("CNN {0}: Validation accuracy={1:.5f}".format(name, history))

