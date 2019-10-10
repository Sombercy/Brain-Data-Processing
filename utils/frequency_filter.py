#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:20:54 2019

@author: savenko
"""
from scipy.fftpack import rfft, irfft, fftfreq
#import numpy as np

def filtering(signal, fmin = None, fmax = None, d = 0.1):
    # FFT filtering
    W = fftfreq(signal.shape[1], d)
    f_signal = rfft(signal)
    cut_signal = []
    # If our original signal time was in seconds, this is now in Hz    
    for fsgn in f_signal:
        cut_f_signal = fsgn.copy()
        if fmin == None: cut_f_signal[(W>fmax)] = 0
        elif fmax == None: cut_f_signal[(W<fmin)] = 0
        else:
            cut_f_signal[(W>fmax)] = 0
            cut_f_signal[(W<fmin)] = 0
        cut_signal.append(irfft(cut_f_signal))
    return cut_signal
    