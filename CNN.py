# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:23:18 2019

@author: Пользователь
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
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler



full_names = ['data-starplus-04847-v7.mat', 'data-starplus-04820-v7.mat', 
         'data-starplus-04799-v7.mat', 'data-starplus-05675-v7.mat',
         'data-starplus-05680-v7.mat', 'data-starplus-05710-v7.mat']
cols = ["CALC", "LIPL", "LT", "LTRIA", "LOPER", "LIPS", "LDLPFC"]

success_indices = []
for indice, full_name in enumerate(full_names):
    fname = 'data-starplus-04820-v7.mat'
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


X_mean = np.mean(X, axis = 3)
fig = plt.figure()
scan = fig.add_subplot(1,1,1)
scan.imshow(scaler.fit_transform(X_mean[0]))
plt.show()
X_train = X_mean.reshape(-1,64,64,1)
Y_train = to_categorical(y, num_classes = 2)

# GLOBAL VARIABLES
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)
styles=[':','-.','--','-',':','-.','--','-',':','-.','--','-']
"""
#EXPERIMENT 1
nets = 3
model = [0] *nets

for j in range(3):
    model[j] = Sequential()
    model[j].add(Conv2D(24,kernel_size=5,padding='same',activation='relu',
            input_shape=(64,64,1)))
    model[j].add(MaxPool2D())
    if j>0:
        model[j].add(Conv2D(48,kernel_size=5,padding='same',activation='relu'))
        model[j].add(MaxPool2D())
    if j>1:
        model[j].add(Conv2D(64,kernel_size=5,padding='same',activation='relu'))
        model[j].add(MaxPool2D(padding='same'))
    model[j].add(Flatten())
    model[j].add(Dense(256, activation='relu'))
    model[j].add(Dense(2, activation='softmax'))
    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#LEAVE ONE OUT CROSS-VALIDATION 
loo = LeaveOneOut()
history = [0] * nets
names = ["(C-P)x1","(C-P)x2","(C-P)x3"]
cv_result = []

for j in range(nets):
    cv_result = []
    for train_index, test_index in loo.split(X_train):
        clf = model[j]
        X_train2, X_val2 = X_train[train_index], X_train[test_index]
        Y_train2, Y_val2 = Y_train[train_index], Y_train[test_index]
        acc = clf.fit(X_train2,Y_train2, batch_size=None, epochs=10, verbose=0,
                      validation_data=(X_val2, Y_val2), workers=1, callbacks=[annealer])
        cv_result.append(acc.history['val_accuracy'])
    history[j] = np.mean(cv_result)
    print("CNN {0}: Validation accuracy={1:.5f}".format(names[j], history[j]))
"""
"""
# CREATE VALIDATION SET
X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.2)
# TRAIN NETWORKS
history = [0] * nets
names = ["(C-P)x1","(C-P)x2","(C-P)x3"]
epochs = 20
for j in range(nets):
    history[j] = model[j].fit(X_train2,Y_train2, batch_size=80, epochs = epochs, 
        validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)
    print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
             names[j],epochs,max(history[j].history['accuracy']),max(history[j].history['val_accuracy']) ))
"""
loo = LeaveOneOut()
#EXPERIMENT 2
nets = 6
model = [0] *nets
for j in range(6):
    model[j] = Sequential()
    model[j].add(Conv2D(j*8+8,kernel_size=5,activation='relu',input_shape=(64,64,1)))
    model[j].add(MaxPool2D())
    model[j].add(Conv2D(j*16+16,kernel_size=5,activation='relu'))
    """
    model[j].add(MaxPool2D())
    model[j].add(Conv2D(j*32+32,kernel_size=5,activation='relu'))
    """
    model[j].add(MaxPool2D())
    model[j].add(Flatten())
    model[j].add(Dense(256, activation='relu'))
    model[j].add(Dense(2, activation='softmax'))
    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])    
    
#LEAVE ONE OUT CROSS-VALIDATION 
history = [0] * nets
names = ["8 maps","16 maps","24 maps","32 maps","48 maps","64 maps"]
cv_result = []

for j in range(nets):
    cv_result = []
    for train_index, test_index in loo.split(X_train):
        clf = model[j]
        X_train2, X_val2 = X_train[train_index], X_train[test_index]
        Y_train2, Y_val2 = Y_train[train_index], Y_train[test_index]
        acc = clf.fit(X_train2,Y_train2, batch_size=None, epochs=5, verbose=0,
                      validation_data=(X_val2, Y_val2), workers=1, callbacks=[annealer])
        cv_result.append(acc.history['val_accuracy'])
    history[j] = np.mean(cv_result)
    print("CNN {0}: Validation accuracy={1:.5f}".format(names[j], history[j]))
    
#EXPERIMENT 3
nets = 4
model = [0] *nets
for j in range(4):
    model[j] = Sequential()
    model[j].add(Conv2D(j*8+8,kernel_size=5,activation='relu',input_shape=(64,64,1)))
    model[j].add(MaxPool2D())
    model[j].add(Conv2D(j*16+16,kernel_size=5,activation='relu'))
    model[j].add(MaxPool2D())
    model[j].add(Conv2D(j*32+32,kernel_size=5,activation='relu'))
    model[j].add(MaxPool2D())
    model[j].add(Flatten())
    model[j].add(Dense(512, activation='relu'))
    model[j].add(Dense(2, activation='softmax'))
    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])    
    
#LEAVE ONE OUT CROSS-VALIDATION 
history = [0] * nets
names = ["8 maps","16 maps","24 maps","32 maps"]
cv_result = []

for j in range(nets):
    cv_result = []
    for train_index, test_index in loo.split(X_train):
        clf = model[j]
        X_train2, X_val2 = X_train[train_index], X_train[test_index]
        Y_train2, Y_val2 = Y_train[train_index], Y_train[test_index]
        acc = clf.fit(X_train2,Y_train2, batch_size=None, epochs=5, verbose=0,
                      validation_data=(X_val2, Y_val2), workers=1, callbacks=[annealer])
        cv_result.append(acc.history['val_accuracy'])
    history[j] = np.mean(cv_result)
    print("CNN {0}: Validation accuracy={1:.5f}".format(names[j], history[j]))
