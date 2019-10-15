#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:20:54 2019

@author: savenko
"""
from scipy.fftpack import rfft, irfft, fftfreq
from scipy.signal import butter, filtfilt
#import numpy as np

def FFT_filter(signal, fmin = None, fmax = None, ss = 0.1):
""" This function cuts the signal to (fmin; fmax) passband. fmin and fmax 
    should be considered in Hz and fs variable respect to signal 
    sampling spacing in seconds."""    
    W = fftfreq(signal.shape[1], ss)
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
    plt.plot(signal, label = 'originals')
    plt.plott(cut_signal, label='filtered')
    plt.legend()
    plt.show()
    return cut_signal
    
def Butterworth_filter(signal, fcut = None, btype = 'low', fs = 10):
""" btype = 'low' or 'hp' """
    # Normalize the frequency
    wcut = fcut / (fs / 2) 
    b, a = signal.butter(5, wcut, btype)
    output = signal.filtfilt(b, a, signal)
    plt.plot(signal, label = 'originals')
    plt.plott( output, label='filtered')
    plt.legend()
    plt.show()
