#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:20:54 2019

@author: savenko
"""
from scipy.fftpack import rfft, irfft, fftfreq
from scipy.signal import butter, lfilter, freqz
import pandas as pd
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing

def FFT_pass(signal, fmin = None, fmax = None, d = 0.1):
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


def butter_bandpass(lowcut, highcut, sampling_rate, order=5):
    nyq_freq = sampling_rate*0.5
    low = lowcut/nyq_freq
    high = highcut/nyq_freq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_high_low_pass(lowcut, highcut, sampling_rate, order=5):
    nyq_freq = sampling_rate*0.5
    lower_bound = lowcut/nyq_freq
    higher_bound = highcut/nyq_freq
    b_high, a_high = butter(order, lower_bound, btype='high')
    b_low, a_low = butter(order, higher_bound, btype='low')
    return b_high, a_high, b_low, a_low

def butter_bandpass_filter(data, lowcut, highcut, sampling_rate, order=5, how_to_filt = 'separately'):
    if how_to_filt == 'separately':
        b_high, a_high, b_low, a_low = butter_high_low_pass(lowcut, highcut, sampling_rate, order=order)
        y = lfilter(b_high, a_high, data)
        y = lfilter(b_low, a_low, y)
    elif how_to_filt == 'simultaneously':
        b, a = butter_bandpass(lowcut, highcut, sampling_rate, order=order)
        y = lfilter(b, a, data)
    return y
"""
# Filter the data, and plot both the original and filtered signals.
X_flt = butter_bandpass_filter(X, 
                              lowcut = 0.53, 
                              highcut = 40, 
                              sampling_rate = 178, 
                              order = 5, 
                              how_to_filt = 'simultaneously')
"""
def rolling_mean(signal, window_size = 15):
    X = pd.DataFrame(signal)
    return X.rolling(window_size, axis = 1).mean().drop([x for x in range(window_size)],
                     axis = 1).values
                     
def exp_smooth(signal, smoothing_lvl = 0.2):
    X = []
    for i in range(signal.shape[0]):
        model = SimpleExpSmoothing(signal[i].copy(order = 'C'))
        X[i] = model.fit(smoothing_level=.2)
    return np.array(X)




