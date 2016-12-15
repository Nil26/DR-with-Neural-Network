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
from rbm import fun_prob_h
from rbm import fun_uphid
from RBM_new import fun_RBM_new

# for fine-tuning
from fine_tuning import fine_tuning

# data construction is mapped from binary data to continous
def autoencoder_output_conti(w_list_4_layer, b_list_4_layer, data, N, num_test):
    # get the autoencoder parameters
    w1 = w_list_4_layer[0]
    w2 = w_list_4_layer[1]
    w3 = w_list_4_layer[2]
    w4 = w_list_4_layer[3]

    hbias1 = np.transpose(b_list_4_layer[0])
    hbias2 = np.transpose(b_list_4_layer[1])
    hbias3 = np.transpose(b_list_4_layer[2])
    hbias4 = np.transpose(b_list_4_layer[3])
    vbias4 = np.transpose(b_list_4_layer[4])
    vbias3 = np.transpose(b_list_4_layer[5])
    vbias2 = np.transpose(b_list_4_layer[6])
    vbias1 = np.transpose(b_list_4_layer[7])

    prob1 = fun_prob_h(data, w1, hbias1, num_test, N[1])
    hid1 = fun_uphid(prob1)
    prob2 = fun_prob_h(hid1, w2, hbias2, num_test, N[2])
    hid2 = fun_uphid(prob2)
    prob3 = fun_prob_h(hid2, w3, hbias3, num_test, N[3])
    hid3 = fun_uphid(prob3)
    prob4 = fun_prob_h(hid3, w4, hbias4, num_test, N[4])
    hid4 = fun_uphid(prob4)
    
    prob5 = fun_prob_d(hid4, w4, vbias4, num_test, N[3])
    vis5 = fun_uphid(prob5)
    prob6 = fun_prob_d(vis5, w3, vbias3, num_test, N[2])
    vis6 = fun_uphid(prob6)
    prob7 = fun_prob_d(vis6, w2, vbias2, num_test, N[1])
    vis7 = fun_uphid(prob7)
    # continuous data reconstruction
    prob8 = fun_prob_d(vis7, w1, vbias1, num_test, N[0])
    data_con = prob8
    
    return hid1, hid2, hid3, hid4, vis5, vis6, vis7, data_con

# data construction is mapped from binary data to binary
def autoencoder_output(w_list_4_layer, b_list_4_layer, data, N, num_test):
    # get the autoencoder parameters
    w1 = w_list_4_layer[0]
    w2 = w_list_4_layer[1]
    w3 = w_list_4_layer[2]
    w4 = w_list_4_layer[3]

    hbias1 = np.transpose(b_list_4_layer[0])
    hbias2 = np.transpose(b_list_4_layer[1])
    hbias3 = np.transpose(b_list_4_layer[2])
    hbias4 = np.transpose(b_list_4_layer[3])
    vbias4 = np.transpose(b_list_4_layer[4])
    vbias3 = np.transpose(b_list_4_layer[5])
    vbias2 = np.transpose(b_list_4_layer[6])
    vbias1 = np.transpose(b_list_4_layer[7])

    prob1 = fun_prob_h(data, w1, hbias1, num_test, N[1])
    hid1 = fun_uphid(prob1)
    prob2 = fun_prob_h(hid1, w2, hbias2, num_test, N[2])
    hid2 = fun_uphid(prob2)
    prob3 = fun_prob_h(hid2, w3, hbias3, num_test, N[3])
    hid3 = fun_uphid(prob3)
    prob4 = fun_prob_h(hid3, w4, hbias4, num_test, N[4])
    hid4 = fun_uphid(prob4)
    
    prob5 = fun_prob_d(hid4, w4, vbias4, num_test, N[3])
    vis5 = fun_uphid(prob5)
    prob6 = fun_prob_d(vis5, w3, vbias3, num_test, N[2])
    vis6 = fun_uphid(prob6)
    prob7 = fun_prob_d(vis6, w2, vbias2, num_test, N[1])
    vis7 = fun_uphid(prob7)
    # binary data reconstruction
    prob8 = fun_prob_d(vis7, w1, vbias1, num_test, N[0])
    data_con = fun_uphid(prob8)
    
    return hid1, hid2, hid3, hid4, vis5, vis6, vis7, data_con
    
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
    
def face_read(n_train):
    yale = sio.loadmat('yalefaces.mat')
    yalefaces = yale['yalefaces']
    # normalize the color data
    face_data = yalefaces.reshape(2016,2414)
    face_norm = face_data/255.
    data = np.transpose(face_norm)
    data_train = data[0:n_train,:]
    data_test = data[n_train:,:]
    return data_train, data_test

#def mandrill_read():
#    #To display this image
#    im = Image.open('mandrill.tiff')
#    #im.show()
#    y = np.array(im)
#    x = y.reshape(262144,3)/255
#    return x
#    
#def pretrain_mandrill_unitest():
#    N = np.array([262144,65536,2])
#    data = mandrill_read()
#    # Do the data treatment(design unit variance) and divide into batches
#    data_batch_train = fun_batch(np.transpose(data),3)
#    # first layer training
#    hid_data, weight_vh, hibias, vibias, E_fun, err = fun_RBM_new(data_batch_train, N[1])
#    
#    # remap the 
#    datacon = fun_prob_d(hid_data[:,:,0], weight_vh, vibias, 3, 262144)    
#    
#    datacon_show = datacon[0,:].reshape(512,512)*255
#    data_show = data_batch_train[0,:,0].reshape(512,512)*255
#    plt.figure()
#    plt.imshow(datacon_show)
#    plt.figure()
#    plt.imshow(data_show)

def pretrain_faces_unitest():
    N = np.array([2016,1400,2])
    data_train, data_test = face_read(2000)
    # Do the data treatment(design unit variance) and divide into batches
    data_batch_train = fun_batch(data_train,5)
    # first layer training
    hid_data, weight_vh, hibias, vibias, E_fun, err = fun_RBM_new(data_batch_train, N[1])
    
    # remap the 
    datacon = fun_prob_d(hid_data[:,:,0], weight_vh, vibias, 400, 2016)    
    
    datacon_show = datacon[1,:].reshape(48,42)*255
    data_show = data_batch_train[1,:,0].reshape(48,42)*255
    plt.imshow(datacon_show)
    plt.imshow(data_show)
                
def RBM_train_unitest_mnist():
    N = np.array([784,1000,2])
    data_train, label_train, data_test, label_test = mnist_read(2000)
    # Do the data treatment(design unit variance) and divide into batches
    data_batch_train = data_treatment(data_train,20)
    # first layer training
    hid_data, weight_vh, hibias, vibias, E_fun, err = fun_RBM(data_batch_train, N[1])
    
    # remap the 
    prob = fun_prob_d(hid_data[:,:,0], weight_vh, vibias, 10, 784)
    datacon = fun_uphid(prob)
    
    plt.figure()
    plt.imshow(datacon[0,:].reshape(28,28))
    plt.figure()
    plt.imshow(data_batch_train[0,:,0].reshape(28,28))

def RBM_train_unitest_new_mnist():
    N = np.array([784,1000,2])
    data_train, label_train, data_test, label_test = mnist_read(2000)
    # Do the data treatment(design unit variance) and divide into batches
    data_batch_train = fun_batch(data_train,20)
    # first layer training
    hid_data, weight_vh, hibias, vibias, E_fun, err = fun_RBM_new(data_batch_train, N[1])
    
    # remap the 
    datacon = fun_prob_d(hid_data[:,:,0], weight_vh, vibias, 10, 784)
    
    plt.figure()
    plt.imshow(datacon[0,:].reshape(28,28))
    plt.figure()
    plt.imshow(data_batch_train[0,:,0].reshape(28,28))
    
# Unitest for doing the fine-tuning process
def fine_tune_faces_unites():
    # constants used in fine-tuning
    training_steps = 200
    num_batch = 100
    lr = 0.01
    
    # read the pretrained weights
    weight_vh1 = np.loadtxt('./pretrain/yale_faces/Weight_matrix_1.txt')
    weight_vh2 = np.loadtxt('./pretrain/yale_faces/Weight_matrix_2.txt')
    weight_vh3 = np.loadtxt('./pretrain/yale_faces/Weight_matrix_3.txt')
    weight_vh4 = np.loadtxt('./pretrain/yale_faces/Weight_matrix_4.txt')
    # construct the fine-tuning input weights
    w_list_4_layer = [weight_vh1, weight_vh2, weight_vh3, weight_vh4]

    # read the pretrained bias
    # hidden bias
    bias1 = np.array([np.loadtxt('./pretrain/yale_faces/Hidden_bias_1.txt')]).transpose()
    bias2 = np.array([np.loadtxt('./pretrain/yale_faces/Hidden_bias_2.txt')]).transpose()
    bias3 = np.array([np.loadtxt('./pretrain/yale_faces/Hidden_bias_3.txt')]).transpose()
    bias4 = np.array([np.loadtxt('./pretrain/yale_faces/Hidden_bias_4.txt')]).transpose()
    # visual bias
    bias5 = np.array([np.loadtxt('./pretrain/yale_faces/Visible_bias_4.txt')]).transpose()
    bias6 = np.array([np.loadtxt('./pretrain/yale_faces/Visible_bias_3.txt')]).transpose()
    bias7 = np.array([np.loadtxt('./pretrain/yale_faces/Visible_bias_2.txt')]).transpose()
    bias8 = np.array([np.loadtxt('./pretrain/yale_faces/Visible_bias_1.txt')]).transpose()
    # construct the fine-tuning input bias
    b_list_4_layer = [bias1, bias2, bias3, bias4, bias5, bias6, bias7, bias8]

    #N = np.array([2016,1400,896,504,224])
    data_train, data_test = face_read(2000)
    data_train_finetune = np.transpose(data_train)
    
    # Do the fine-tuning
    total_cost_record_4_lay, w_list_output_4_lay, b_list_output_4_lay = fine_tuning(w_list_4_layer, b_list_4_layer, data_train_finetune, num_batch, training_steps, lr)
    
    # write the weight matrix
    for i in np.arange(4):
        filename = 'W_matrix_'+str(i+1)+'_Final.txt'
        np.savetxt(filename, w_list_output_4_lay[i])
    
    # write the bias    
    for j in np.arange(8):
        filename = 'B_bias_'+str(j+1)+'_Final.txt'
        np.savetxt(filename, b_list_output_4_lay[j])
        
    # write the total cost
    np.savetxt('Total_Cost_Final.txt', total_cost_record_4_lay)
    
def fine_tune_mnist_unites():
    # constants used in fine-tuning
    training_steps = 200
    num_batch = 100
    lr = 0.01
    
    # read the pretrained weights
    weight_vh1 = np.loadtxt('./pretrain/mnist/Weight_matrix_1.txt')
    weight_vh2 = np.loadtxt('./pretrain/mnist/Weight_matrix_2.txt')
    weight_vh3 = np.loadtxt('./pretrain/mnist/Weight_matrix_3.txt')
    weight_vh4 = np.loadtxt('./pretrain/mnist/Weight_matrix_4.txt')
    # construct the fine-tuning input weights
    w_list_4_layer = [weight_vh1, weight_vh2, weight_vh3, weight_vh4]

    # read the pretrained bias
    # hidden bias
    bias1 = np.array([np.loadtxt('./pretrain/mnist/Hidden_bias_1.txt')]).transpose()
    bias2 = np.array([np.loadtxt('./pretrain/mnist/Hidden_bias_2.txt')]).transpose()
    bias3 = np.array([np.loadtxt('./pretrain/mnist/Hidden_bias_3.txt')]).transpose()
    bias4 = np.array([np.loadtxt('./pretrain/mnist/Hidden_bias_4.txt')]).transpose()
    # visual bias
    bias5 = np.array([np.loadtxt('./pretrain/mnist/Visible_bias_4.txt')]).transpose()
    bias6 = np.array([np.loadtxt('./pretrain/mnist/Visible_bias_3.txt')]).transpose()
    bias7 = np.array([np.loadtxt('./pretrain/mnist/Visible_bias_2.txt')]).transpose()
    bias8 = np.array([np.loadtxt('./pretrain/mnist/Visible_bias_1.txt')]).transpose()
    # construct the fine-tuning input bias
    b_list_4_layer = [bias1, bias2, bias3, bias4, bias5, bias6, bias7, bias8]

    #N = np.array([784,1000,500,250,2])
    data_train, label_train, data_test, label_test = mnist_read(2000)
    data_train_finetune = np.transpose(data_train)
    
    # Do the fine-tuning
    total_cost_record_4_lay, w_list_output_4_lay, b_list_output_4_lay = fine_tuning(w_list_4_layer, b_list_4_layer, data_train_finetune, num_batch, training_steps, lr)
    
    # write the weight matrix
    for i in np.arange(4):
        filename = 'W_matrix_'+str(i+1)+'_Final.txt'
        np.savetxt(filename, w_list_output_4_lay[i])
    
    # write the bias    
    for j in np.arange(8):
        filename = 'B_bias_'+str(j+1)+'_Final.txt'
        np.savetxt(filename, b_list_output_4_lay[j])
        
    # write the total cost
    np.savetxt('Total_Cost_Final.txt', total_cost_record_4_lay)
    
def result_faces():
    num_train = 2000
    num_test = 2414 - num_train
    N = np.array([2016,1400,896,504,224])
    data_train, data_test = face_read(num_train)
    
#    # Read the data for weight matrix and bias
#    # read the pretrained weights
#    weight_vh1 = np.loadtxt('./fine-tuning/yale_faces/W_matrix_1_Final.txt')
#    weight_vh2 = np.loadtxt('./fine-tuning/yale_faces/W_matrix_2_Final.txt')
#    weight_vh3 = np.loadtxt('./fine-tuning/yale_faces/W_matrix_3_Final.txt')
#    weight_vh4 = np.loadtxt('./fine-tuning/yale_faces/W_matrix_4_Final.txt')
#    # construct the fine-tuning input weights
#    w_list_4_layer = [weight_vh1, weight_vh2, weight_vh3, weight_vh4]
#
#    # read the pretrained bias
#    # hidden bias
#    bias1 = np.array([np.loadtxt('./fine-tuning/yale_faces/B_bias_1_Final.txt')]).transpose()
#    bias2 = np.array([np.loadtxt('./fine-tuning/yale_faces/B_bias_2_Final.txt')]).transpose()
#    bias3 = np.array([np.loadtxt('./fine-tuning/yale_faces/B_bias_3_Final.txt')]).transpose()
#    bias4 = np.array([np.loadtxt('./fine-tuning/yale_faces/B_bias_4_Final.txt')]).transpose()
#    # visual bias
#    bias5 = np.array([np.loadtxt('./fine-tuning/yale_faces/B_bias_5_Final.txt')]).transpose()
#    bias6 = np.array([np.loadtxt('./fine-tuning/yale_faces/B_bias_6_Final.txt')]).transpose()
#    bias7 = np.array([np.loadtxt('./fine-tuning/yale_faces/B_bias_7_Final.txt')]).transpose()
#    bias8 = np.array([np.loadtxt('./fine-tuning/yale_faces/B_bias_8_Final.txt')]).transpose()
#    # construct the fine-tuning input bias
#    b_list_4_layer = [bias1, bias2, bias3, bias4, bias5, bias6, bias7, bias8]


    #RBM result
    # read the pretrained weights
    weight_vh1 = np.loadtxt('./pretrain/yale_faces/Weight_matrix_1.txt')
    weight_vh2 = np.loadtxt('./pretrain/yale_faces/Weight_matrix_2.txt')
    weight_vh3 = np.loadtxt('./pretrain/yale_faces/Weight_matrix_3.txt')
    weight_vh4 = np.loadtxt('./pretrain/yale_faces/Weight_matrix_4.txt')
    # construct the fine-tuning input weights
    w_list_4_layer = [weight_vh1, weight_vh2, weight_vh3, weight_vh4]

    # read the pretrained bias
    # hidden bias
    bias1 = np.array([np.loadtxt('./pretrain/yale_faces/Hidden_bias_1.txt')]).transpose()
    bias2 = np.array([np.loadtxt('./pretrain/yale_faces/Hidden_bias_2.txt')]).transpose()
    bias3 = np.array([np.loadtxt('./pretrain/yale_faces/Hidden_bias_3.txt')]).transpose()
    bias4 = np.array([np.loadtxt('./pretrain/yale_faces/Hidden_bias_4.txt')]).transpose()
    # visual bias
    bias5 = np.array([np.loadtxt('./pretrain/yale_faces/Visible_bias_4.txt')]).transpose()
    bias6 = np.array([np.loadtxt('./pretrain/yale_faces/Visible_bias_3.txt')]).transpose()
    bias7 = np.array([np.loadtxt('./pretrain/yale_faces/Visible_bias_2.txt')]).transpose()
    bias8 = np.array([np.loadtxt('./pretrain/yale_faces/Visible_bias_1.txt')]).transpose()
    # construct the fine-tuning input bias
    b_list_4_layer = [bias1, bias2, bias3, bias4, bias5, bias6, bias7, bias8]

    hid1, hid2, hid3, hid4, vis5, vis6, vis7, data_con = autoencoder_output_conti(w_list_4_layer, b_list_4_layer, data_train, N, num_train)

    figure_index = 113
    
    datacon_show = data_con[figure_index,:].reshape(48,42)*255
    data_show = data_test[figure_index,:].reshape(48,42)*255
    plt.figure()
    plt.imshow(datacon_show, vmin = 0, vmax = 255)
    plt.figure()
    plt.imshow(data_show, vmin = 0, vmax = 255)
        
def result_mnist():
    num_train = 2000
    num_test = 3000 - num_train
    N = np.array([784,1000,500,250,2]) 
    data_train, label_train, data_test, label_test = mnist_read(num_train)
    
#    # Read the data for weight matrix and bias
#    # read the pretrained weights
#    weight_vh1 = np.loadtxt('./fine-tuning/mnist/W_matrix_1_Final.txt')
#    weight_vh2 = np.loadtxt('./fine-tuning/mnist/W_matrix_2_Final.txt')
#    weight_vh3 = np.loadtxt('./fine-tuning/mnist/W_matrix_3_Final.txt')
#    weight_vh4 = np.loadtxt('./fine-tuning/mnist/W_matrix_4_Final.txt')
#    # construct the fine-tuning input weights
#    w_list_4_layer = [weight_vh1, weight_vh2, weight_vh3, weight_vh4]
#
#    # read the pretrained bias
#    # hidden bias
#    bias1 = np.array([np.loadtxt('./fine-tuning/mnist/B_bias_1_Final.txt')]).transpose()
#    bias2 = np.array([np.loadtxt('./fine-tuning/mnist/B_bias_2_Final.txt')]).transpose()
#    bias3 = np.array([np.loadtxt('./fine-tuning/mnist/B_bias_3_Final.txt')]).transpose()
#    bias4 = np.array([np.loadtxt('./fine-tuning/mnist/B_bias_4_Final.txt')]).transpose()
#    # visual bias
#    bias5 = np.array([np.loadtxt('./fine-tuning/mnist/B_bias_5_Final.txt')]).transpose()
#    bias6 = np.array([np.loadtxt('./fine-tuning/mnist/B_bias_6_Final.txt')]).transpose()
#    bias7 = np.array([np.loadtxt('./fine-tuning/mnist/B_bias_7_Final.txt')]).transpose()
#    bias8 = np.array([np.loadtxt('./fine-tuning/mnist/B_bias_8_Final.txt')]).transpose()
#    # construct the fine-tuning input bias
#    b_list_4_layer = [bias1, bias2, bias3, bias4, bias5, bias6, bias7, bias8]


    #RBM result
    # read the pretrained weights
    weight_vh1 = np.loadtxt('./pretrain/mnist/Weight_matrix_1.txt')
    weight_vh2 = np.loadtxt('./pretrain/mnist/Weight_matrix_2.txt')
    weight_vh3 = np.loadtxt('./pretrain/mnist/Weight_matrix_3.txt')
    weight_vh4 = np.loadtxt('./pretrain/mnist/Weight_matrix_4.txt')
    # construct the fine-tuning input weights
    w_list_4_layer = [weight_vh1, weight_vh2, weight_vh3, weight_vh4]

    # read the pretrained bias
    # hidden bias
    bias1 = np.array([np.loadtxt('./pretrain/mnist/Hidden_bias_1.txt')]).transpose()
    bias2 = np.array([np.loadtxt('./pretrain/mnist/Hidden_bias_2.txt')]).transpose()
    bias3 = np.array([np.loadtxt('./pretrain/mnist/Hidden_bias_3.txt')]).transpose()
    bias4 = np.array([np.loadtxt('./pretrain/mnist/Hidden_bias_4.txt')]).transpose()
    # visual bias
    bias5 = np.array([np.loadtxt('./pretrain/mnist/Visible_bias_4.txt')]).transpose()
    bias6 = np.array([np.loadtxt('./pretrain/mnist/Visible_bias_3.txt')]).transpose()
    bias7 = np.array([np.loadtxt('./pretrain/mnist/Visible_bias_2.txt')]).transpose()
    bias8 = np.array([np.loadtxt('./pretrain/mnist/Visible_bias_1.txt')]).transpose()
    # construct the fine-tuning input bias
    b_list_4_layer = [bias1, bias2, bias3, bias4, bias5, bias6, bias7, bias8]

    hid1, hid2, hid3, hid4, vis5, vis6, vis7, data_con = autoencoder_output(w_list_4_layer, b_list_4_layer, data_train, N, num_train)

    figure_index = 21
    
    datacon_show = data_con[figure_index,:].reshape(28,28)
    data_show = data_test[figure_index,:].reshape(28,28)
    plt.figure()
    plt.imshow(datacon_show)
    plt.figure()
    plt.imshow(data_show)
    
    # map from the first hidden layer
    datacon_1stlayer = fun_prob_d(hid1[figure_index,:], weight_vh1, bias8, 1, 784)
    plt.figure()
    plt.imshow(datacon_1stlayer)
    