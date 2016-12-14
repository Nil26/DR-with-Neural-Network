#!/usr/bin/env python3
"""
@author: zhshang
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image

# data treatment
from data_treatment import data_treatment
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

N = np.array([784,1000,500,250,2])
data_train, label_train, data_test, label_test = mnist_read(2000)
# Do the data treatment(design unit variance) and divide into batches
data_batch_train = data_treatment(data_train,100)
# first layer training
hid_data1, weight_vh1, hibias1, vibias1, E_fun1, err1 = fun_RBM(data_batch_train, N[1])

# second layer
hid_data2, weight_vh2, hibias2, vibias2, E_fun2, err2 = fun_RBM(hid_data1, N[2])

# third layer
hid_data3, weight_vh3, hibias3, vibias3, E_fun3, err3 = fun_RBM(hid_data2, N[3])

# fourth layer
hid_data4, weight_vh4, hibias4, vibias4, E_fun4, err4 = fun_RBM(hid_data3, N[4])

np.savetxt('Hidden_layer_1.txt',np.transpose(hid_data1,(0,2,1)).reshape(2000, N[1]))
np.savetxt('Weight_matrix_1.txt',weight_vh1)
np.savetxt('Hidden_bias_1.txt',hibias1)
np.savetxt('Visible_bias_1.txt',vibias1)
np.savetxt('Energy_function_1.txt',E_fun1)
np.savetxt('Err_1.txt',err1)

np.savetxt('Hidden_layer_2.txt',np.transpose(hid_data2,(0,2,1)).reshape(2000, N[2]))
np.savetxt('Weight_matrix_2.txt',weight_vh2)
np.savetxt('Hidden_bias_2.txt',hibias2)
np.savetxt('Visible_bias_2.txt',vibias2)
np.savetxt('Energy_function_2.txt',E_fun2)
np.savetxt('Err_2.txt',err2)

np.savetxt('Hidden_layer_3.txt',np.transpose(hid_data3,(0,2,1)).reshape(2000, N[3]))
np.savetxt('Weight_matrix_3.txt',weight_vh3)
np.savetxt('Hidden_bias_3.txt',hibias3)
np.savetxt('Visible_bias_3.txt',vibias3)
np.savetxt('Energy_function_3.txt',E_fun3)
np.savetxt('Err_3.txt',err3)

np.savetxt('Hidden_layer_4.txt',np.transpose(hid_data4,(0,2,1)).reshape(2000, N[4]))
np.savetxt('Weight_matrix_4.txt',weight_vh4)
np.savetxt('Hidden_bias_4.txt',hibias4)
np.savetxt('Visible_bias_4.txt',vibias4)
np.savetxt('Energy_function_4.txt',E_fun4)
np.savetxt('Err_4.txt',err4)
