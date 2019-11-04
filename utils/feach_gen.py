#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:01:00 2019

@author: savenko
"""

import numpy as np
from scipy.sparse import diags
from scipy.special import gamma
from scipy.integrate import simps
from itertools import chain
from scipy.fftpack import rfft

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

def gen_eigen(X, channel = None):
    X -= X.min().min()
    lamda = []
    Nh = []
    if channel == 'single':
        for i in range(len(X)):
            y = np.array(X[i])
            h = max(y)/20
            yscsa, kappa, n, psinnor = scsa(y, h)
            lamda.append(kappa.diagonal().tolist())
            Nh.append(n)
        N = min(Nh)
        return np.array([lamda[i][:N] for i in range(len(lamda))])
    elif channel == 'multi':
        for i in range(len(X)):
            eig = []
            eig_n = []
            for j in range(X[i].shape[1]):
                y = np.array(X[i][j])
                h = max(y)/20
                yscsa, kappa, n, psinnor = scsa(y, h)
                eig.append(kappa.diagonal().tolist())
                eig_n.append(n)
            lamda.append(eig)
            Nh.append(eig_n)
        N = [min(np.asarray(Nh)[:, j]) for j in range(X.shape[2])]
        return np.array([list(chain.from_iterable([lamda[i][j][:N[j]] for j in range(len(N))])) for i in range(len(lamda))])
    
def avactivity(X, t0, tmin, tmax, sfreq):
    
    """This function calculate average activity of each channel during a 
       given period (tmin, tmax) and sampling frequency sfreq of the signal. 
       After that concatenation of means is done.
       For multi-channel data with shape (samples, channels, time) only"""
    
    begin = np.round((tmin - t0) * sfreq).astype(np.int)
    end = np.round((tmax - t0) * sfreq).astype(np.int)
    data = X[:, :, begin:end].copy()
    data = np.mean(data, axis = 1)
    X = np.array([list(chain.from_iterable(data[i]))for i in range(len(data))])
    return X

def major_freqs(signal, channel, k_first = None):
    
    """This function returns k_first dominant frequencies of signal spectrum"""
    if channel == 'single':
        y = np.array([abs(rfft(signal[i])) for i in range(len(signal))])
        if k_first == None:
            X = y.argsort()[:][::-1]
        else:
            X = y.argsort()[-k_first:][::-1]
    elif channel == 'multi':
        y = [abs(rfft(list(chain.from_iterable(signal[i])))) for i in range(len(signal))]
        X = y.argsort()[:][::-1]
    return X
