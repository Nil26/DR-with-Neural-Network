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
from data_treatment import fun_batch
# rbm first layer
from rbmc_b import fun_RBM_con
# rbm following layers
from rbm import fun_RBM

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

def AE_forward(X, Wb, N, batch_size):
    d, n = X.shape
                
def train_AE():
    num_layers = np.size(N)
    data_train, label_train, data_test, label_test = mnist_read(2000)
    # Do the data treatment(design unit variance) and divide into batches
    data_batch_train = data_treatment(data_train,20)
    # first layer training
    hid_data, weight_vh, hibias, vibias, E_fun = fun_RBM_con(data_batch_train, N[1])
    return data_batch_train
    
def RBM_train_unitest():
    N = np.array([784,400,2])
    data_train, label_train, data_test, label_test = mnist_read(100)
    # Do the data treatment(design unit variance) and divide into batches
    data_batch_train = fun_batch(data_train,10)
    # first layer training
    hid_data, weight_vh, hibias, vibias, E_fun = fun_RBM(data_batch_train, N[1])
    
    
def plot_mnist(train_data, train_labels, test_data, test_labels, Wb, N, RBM_error, BP_error):
    '''
    function plot_mnist ( train_data, train_labels, test_data, test_labels, ...
    Wb, n, RBM_error, BP_error )
    This function displays the results of the autoencoder on the MNIST
    data, assuming an encoder with final dimension 2.  The results are
    displayed using two dimensional plots and plots of reconstruction error
    at the different phases of the algorithm.

    INPUTS: train_data, test_data -- input data with points as columns.
    train_labals, test_labels -- labels of corresponding data.
    Wb, n -- autoencoder weights, biases, and dimensions.
    RBM_error, BP_error -- error matrices as returned by train_AE.

    OUTPUTS: Error plots are shown in figures 1 & 2, visualizations are
    shown in figures 3.
    '''    
    # RBM error plot in figure 1
    plt.figure()
    plt.clf()
    plt.plot(RBM_error)
    title('Reconstruction Error for RBM pre-training')
    xlabel('Iteration')
    ylabel('Mean Squared Error')
    
    # BP error plot in figure 2
    plt.figure()
    plt.clf()
    plt.plot(BP_error)
    title('Reconstruction Error for fine-tuning')
    xlabel('Iteration')
    ylabel('Mean Squared Error')
    
    # mnist visualization - training data
    Y = AE_forward(train_data, Wb, n, batch_size)
    plt.figure()
    plt.clf()
    plt.plot(Y)
    title('Training Data')
    
    # mnist visualization - test data
    Y = AE_forward(test_data, Wb, n, batch_size)
    plt.figure()
    plt.clf()
    plt.plot(Y)
    title('Training Data')