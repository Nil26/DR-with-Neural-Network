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
    data_batch = np.zeros((numcases, numbatches, d))
    #random assaigned to batches
    random.shuffle(data)
    data_batch = data.reshape((numcases, numbatches, d))
    return data_batch.transpose((0,2,1))
    
def data_treatment(data, numbatches):
    n, d = data.shape
    numcases = n / numbatches
    #initialize
    data_new = np.zeros((n,d))
    data_batch = np.zeros((numcases, d, numbatches))
    data_new = fun_binary(data)
    data_batch = fun_batch(data_new, numbatches)
    return data_batch
