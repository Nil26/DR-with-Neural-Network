#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 15:43:51 2016

@author: zhshang
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image

# data treatment
from data_treatment import data_treatment
from data_treatment import fun_batch
# rbm following layers
from rbm import fun_RBM

from rbm import fun_prob_d
from rbm import fun_uphid
from RBM_new import fun_RBM_new

def mnist_read(n_train):
    mnist_49_3000 = sio.loadmat('mnist_49_3000.mat')
    x = mnist_49_3000['x']
    y = mnist_49_3000['y']
    d,n= x.shape
    data = np.transpose(x)
    label = np.transpose(y)
    data_train = data[0:n_train,:]
    label_train = label[0:n_train,:]
    data_test = data[n_train:,:]
    label_test = label[n_train:,:]
    return data_train, label_train, data_test, label_test

def yalefaces_read():
    yale = sio.loadmat('yalefaces.mat')
    yalefaces = yale['yalefaces']
    for i in range(0,yalefaces.shape[2]):
        x = yalefaces[:,:,i]
    ax.imshow(x, extent=[0, 1, 0, 1])
    plt.imshow(x, cmap=plt.get_cmap('gray'))
    plt.show()

def mandrill_read():
    #To display this image
    im = Image.open('mandrill.tiff')
    #im.show()
    y = np.array(im)
    M = 2 #block side-length
    n = np.prod(y.shape)/(3*M*M) #number of blocks
    d = y.shape[0]
    c = 0 # counter
    x = np.zeros([n,3*M*M])
    for i in range(0,d,M):
            for j in range(0,d,M):
                #print c,i,j,M,y[i:i+M,j:j+M,:].shape, M*M*3
                x[c,:] = np.reshape(y[i:i+M,j:j+M,:],[1,M*M*3])
                c = c+1                
    
def RBM_train_unitest_mnist():
    N = np.array([784,400,2])
    data_train, label_train, data_test, label_test = mnist_read(100)
    # Do the data treatment(design unit variance) and divide into batches
    data_batch_train = data_treatment(data_train,10)
    # first layer training
    hid_data, weight_vh, hibias, vibias, E_fun, err = fun_RBM(data_batch_train, N[1])
    
    # remap the 
    prob = fun_prob_d(hid_data[:,:,0], weight_vh, vibias, 10, 784)
    datacon = fun_uphid(prob)
    
    plt.imshow(datacon[0,:].reshape(28,28))
    plt.imshow(data_batch_train[0,:,0].reshape(28,28))

def RBM_train_unitest_new_mnist():
    N = np.array([784,400,2])
    data_train, label_train, data_test, label_test = mnist_read(100)
    # Do the data treatment(design unit variance) and divide into batches
    data_batch_train = fun_batch(data_train,10)
    # first layer training
    hid_data, weight_vh, hibias, vibias, E_fun, err = fun_RBM_new(data_batch_train, N[1])
    
    # remap the 
    datacon = fun_prob_d(hid_data[:,:,0], weight_vh, vibias, 10, 784)
    
    plt.imshow(datacon[0,:].reshape(28,28))
    plt.imshow(data_batch_train[0,:,0].reshape(28,28))
    
