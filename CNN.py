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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from scipy.fftpack import fft



full_names = ['data-starplus-04847-v7.mat', 'data-starplus-04820-v7.mat', 
         'data-starplus-04799-v7.mat', 'data-starplus-05675-v7.mat',
         'data-starplus-05680-v7.mat', 'data-starplus-05710-v7.mat']
cols = ["CALC", "LIPL", "LT", "LTRIA", "LOPER", "LIPS", "LDLPFC"]

success_indices = []
for indice, full_name in enumerate(full_names):
    print("Fetching file : %s" % full_name)
    # General information
    data = io.loadmat(full_name)
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
            ROI[x, y, z] = np.nan
        
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
    y = fig.add_subplot(2,4,num+1)
    y.imshow(scaler.fit_transform(X[0][:,:,num]))
plt.show()

#plot the first image in the dataset
#plt.imshow(X[0])