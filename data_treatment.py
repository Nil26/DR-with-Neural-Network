#!/usr/bin/python
import numpy as np
import numpy.random as random

#data treatment

def fun_binary(data):
    n, d = data.shape
    #initialize
    data_new = np.zeros((n, d))
    #normalization
    data_new = np.rint(data)    
    return data_new

def fun_batch(data, numbatches):
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
    data_new = fun_binary(data)
    data_batch = fun_batch(data_new, numbatches)
    return data_batch
    
    
def fun_normal(data):
    n, d = data.shape
    #initialize
    data_new = np.zeros((n, d))
    sigma = np.array([np.zeros((1,d))])
    mean = np.zeros((1,d))
    #normalization
    sigma = np.array([np.std(data, axis=0)])
    mean = np.array([np.mean(data, axis=0)])
    sigma_nonzero = np.add(sigma, (sigma == 0))
    data_new = np.divide(np.subtract(data, np.repeat(mean, n, axis=0)), np.repeat(sigma_nonzero, n, axis=0))   
    return data_new

def data_treatment_new(data, numbatches):
    n, d = data.shape
    numcases = n / numbatches
    #initialize
    data_new = np.zeros((n,d))
    data_batch = np.zeros((numcases, d, numbatches))
    data_new = fun_normal(data)
    data_batch = fun_batch(data_new, numbatches)
    return data_batch