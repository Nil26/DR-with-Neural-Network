#!/usr/bin/python
import numpy as np
import numpy.random as random

#data treatment

def fun_normal(data):
    n, d = data.shape
    #initialize
    sigma = np.zeros((1,d))
    data_new = np.zeros((n, d))
    #normalization
    sigma = np.array([np.std(data, 0)])
    sigma_nonzero = np.add(sigma, (sigma==0))
    data_new = np.divide(data, np.repeat(sigma_nonzero, n, axis=0))
    return data_new

def fun_batch(numbatches, data):
    #initialize
    n,d = data.shape
    numcases = n / numbatches
    data_batch = np.zeros((numcases, d, numbatches))
    #random assaigned to batches
    random.shuffle(data)
    data_batch = data.reshape((numcases, d, numbatches))
    return data_batch
    
def data_treatment(data, numbatches):
    n, d = data.shape
    numcases = n / numbatches
    #initialize
    data_new = np.zeros((n,d))
    data_batch = np.zeros((numcases, d, numbatches))
    data_new = fun_normal(data)
    data_batch = fun_batch(numbatches, data_new)
    return data_batch
