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

    hbias1 = b_list_4_layer[0]
    hbias2 = b_list_4_layer[1]
    hbias3 = b_list_4_layer[2]
    hbias4 = b_list_4_layer[3]
    vbias4 = b_list_4_layer[4]
    vbias3 = b_list_4_layer[5]
    vbias2 = b_list_4_layer[6]
    vbias1 = b_list_4_layer[7]

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
    
    return hid1, hid2, hid3, hid4, vis5, vis6, vis7, data_con, prob4

def autoencoder_output_conti_final(w_list_4_layer, b_list_4_layer, data, N, num_test):
    # get the autoencoder parameters
    w1 = w_list_4_layer[0]
    w2 = w_list_4_layer[1]
    w3 = w_list_4_layer[2]
    w4 = w_list_4_layer[3]
    w5 = w_list_4_layer[4]
    w6 = w_list_4_layer[5]
    w7 = w_list_4_layer[6]
    w8 = w_list_4_layer[7]

    hbias1 = b_list_4_layer[0]
    hbias2 = b_list_4_layer[1]
    hbias3 = b_list_4_layer[2]
    hbias4 = b_list_4_layer[3]
    vbias4 = b_list_4_layer[4]
    vbias3 = b_list_4_layer[5]
    vbias2 = b_list_4_layer[6]
    vbias1 = b_list_4_layer[7]

    prob1 = fun_prob_h(data, w1, hbias1, num_test, N[1])
    hid1 = fun_uphid(prob1)
    prob2 = fun_prob_h(hid1, w2, hbias2, num_test, N[2])
    hid2 = fun_uphid(prob2)
    prob3 = fun_prob_h(hid2, w3, hbias3, num_test, N[3])
    hid3 = fun_uphid(prob3)
    prob4 = fun_prob_h(hid3, w4, hbias4, num_test, N[4])
    hid4 = fun_uphid(prob4)
    
    prob5 = fun_prob_d(hid4, np.transpose(w5), vbias4, num_test, N[3])
    vis5 = fun_uphid(prob5)
    prob6 = fun_prob_d(vis5, np.transpose(w6), vbias3, num_test, N[2])
    vis6 = fun_uphid(prob6)
    prob7 = fun_prob_d(vis6, np.transpose(w7), vbias2, num_test, N[1])
    vis7 = fun_uphid(prob7)
    # continuous data reconstruction
    prob8 = fun_prob_d(vis7, np.transpose(w8), vbias1, num_test, N[0])
    data_con = prob8
    
    return hid1, hid2, hid3, hid4, vis5, vis6, vis7, data_con, prob4

    
# data construction is mapped from binary data to binary
def autoencoder_output(w_list_4_layer, b_list_4_layer, data, N, num_test):
    # get the autoencoder parameters
    w1 = w_list_4_layer[0]
    w2 = w_list_4_layer[1]
    w3 = w_list_4_layer[2]
    w4 = w_list_4_layer[3]

    hbias1 = b_list_4_layer[0]
    hbias2 = b_list_4_layer[1]
    hbias3 = b_list_4_layer[2]
    hbias4 = b_list_4_layer[3]
    vbias4 = b_list_4_layer[4]
    vbias3 = b_list_4_layer[5]
    vbias2 = b_list_4_layer[6]
    vbias1 = b_list_4_layer[7]

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
    data_con = fun_uphid(prob8)
    
    return hid1, hid2, hid3, hid4, vis5, vis6, vis7, data_con, prob4
    
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
    
    #datacon = fun_prob_d(hid_data[:,:,0], weight_vh, vibias, 400, 2016)    
    
    #datacon_show = datacon[0,:].reshape(48,42)*255
    #data_show = data_batch_train[0,:,0].reshape(48,42)*255
    #plt.imshow(datacon_show)
    #plt.imshow(data_show)
    #!!!!!!!!!!!!!!!!!
    num_train = 2000
    N = np.array([2016,1400,896,504,224])
    data_train, data_test = face_read(2000)
    # Do the data treatment(design unit variance) and divide into batches
    data_batch_train = fun_batch(data_train,20)
    # first layer training
    hid_data1, weight_vh1, hibias1, vibias1, E_fun1, err1 = fun_RBM_new(data_batch_train, N[1])
    
    # second layer training
    hid_data2, weight_vh2, hibias2, vibias2, E_fun2, err2 = fun_RBM_new(hid_data1, N[2])
    
    # third layer training
    hid_data3, weight_vh3, hibias3, vibias3, E_fun3, err3 = fun_RBM_new(hid_data2, N[3])
    
    # fourth layer training
    hid_data4, weight_vh4, hibias4, vibias4, E_fun4, err4 = fun_RBM_new(hid_data3, N[4])
    
    w_list_4_layer = [weight_vh1, weight_vh2, weight_vh3, weight_vh4]
    b_list_4_layer = [hibias1, hibias2, hibias3, hibias4, vibias4, vibias3, vibias2, vibias1]

    #print out
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
    
    hid1, hid2, hid3, hid4, vis5, vis6, vis7, data_con, map2d = autoencoder_output_conti(w_list_4_layer, b_list_4_layer, data_train, N, num_train)
    
                
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
    
    plt.figure()
    plt.imshow(datacon[0,:].reshape(28,28))
    plt.figure()
    plt.imshow(data_batch_train[0,:,0].reshape(28,28))

def RBM_train_unitest_new_mnist():
    num_train = 2000
    N = np.array([784,1000,500,250,2])
    data_train, label_train, data_test, label_test = mnist_read(num_train)
    # Do the data treatment(design unit variance) and divide into batches
    data_batch_train = fun_batch(data_train,20)
    # first layer training
    hid_data1, weight_vh1, hibias1, vibias1, E_fun1, err1 = fun_RBM_new(data_batch_train, N[1])
    
    # second layer training
    hid_data2, weight_vh2, hibias2, vibias2, E_fun2, err2 = fun_RBM_new(hid_data1, N[2])
    
    # third layer training
    hid_data3, weight_vh3, hibias3, vibias3, E_fun3, err3 = fun_RBM_new(hid_data2, N[3])
    
    # fourth layer training
    hid_data4, weight_vh4, hibias4, vibias4, E_fun4, err4 = fun_RBM_new(hid_data3, N[4])
    
    w_list_4_layer = [weight_vh1, weight_vh2, weight_vh3, weight_vh4]
    b_list_4_layer = [hibias1, hibias2, hibias3, hibias4, vibias4, vibias3, vibias2, vibias1]

    #print out
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
    
    hid1, hid2, hid3, hid4, vis5, vis6, vis7, data_con, map2d = autoencoder_output_conti(w_list_4_layer, b_list_4_layer, data_train, N, num_train)
    
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
    weight_vh1_ft = np.loadtxt('./Weight_matrix_1.txt')
    weight_vh2 = np.loadtxt('./pretrain/mnist/Weight_matrix_2.txt')
    weight_vh3 = np.loadtxt('./pretrain/mnist/Weight_matrix_3.txt')
    weight_vh4 = np.loadtxt('./pretrain/mnist/Weight_matrix_4.txt')
    # construct the fine-tuning input weights
    w_list_4_layer = [weight_vh1, weight_vh2, weight_vh3, weight_vh4]

    # read the pretrained bias
    # hidden bias
    bias1_ft = np.array([np.loadtxt('./Hidden_bias_1.txt')]).transpose()
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
    weight_vh1 = np.loadtxt('./pretrain_20batch_pc_faces/Weight_matrix_1.txt')
    weight_vh2 = np.loadtxt('./pretrain_20batch_pc_faces/Weight_matrix_2.txt')
    weight_vh3 = np.loadtxt('./pretrain_20batch_pc_faces/Weight_matrix_3.txt')
    weight_vh4 = np.loadtxt('./pretrain_20batch_pc_faces/Weight_matrix_4.txt')
    # construct the fine-tuning input weights
    w_list_4_layer = [weight_vh1, weight_vh2, weight_vh3, weight_vh4]

    # read the pretrained bias
    # hidden bias
    bias1 = [np.loadtxt('./pretrain_20batch_pc_faces/Hidden_bias_1.txt')]
    bias2 = [np.loadtxt('./pretrain_20batch_pc_faces/Hidden_bias_2.txt')]
    bias3 = [np.loadtxt('./pretrain_20batch_pc_faces/Hidden_bias_3.txt')]
    bias4 = [np.loadtxt('./pretrain_20batch_pc_faces/Hidden_bias_4.txt')]
    # visual bias
    bias5 = [np.loadtxt('./pretrain_20batch_pc_faces/Visible_bias_4.txt')]
    bias6 = [np.loadtxt('./pretrain_20batch_pc_faces/Visible_bias_3.txt')]
    bias7 = [np.loadtxt('./pretrain_20batch_pc_faces/Visible_bias_2.txt')]
    bias8 = [np.loadtxt('./pretrain_20batch_pc_faces/Visible_bias_1.txt')]
    # construct the fine-tuning input bias
    b_list_4_layer = [bias1, bias2, bias3, bias4, bias5, bias6, bias7, bias8]
    
    # for pre-train
    hid1_train, hid2_train, hid3_train, hid4_train, vis5_train, vis6_train, vis7_train, data_con_train, map2d_train = autoencoder_output_conti(w_list_4_layer, b_list_4_layer, data_train, N, num_train)
    hid1_test, hid2_test, hid3_test, hid4_test, vis5_test, vis6_test, vis7_test, data_con_test, map2d_test = autoencoder_output_conti(w_list_4_layer, b_list_4_layer, data_test, N, num_test)
    
    # index number
    figure_index_train = np.rint(np.random.rand()*num_train)
    figure_index_test = np.rint(np.random.rand()*num_test)
    
    # compare with training data
    datacon_show = data_con_train[figure_index_train,:].reshape(48,42)*255
    data_show = data_train[figure_index_train,:].reshape(48,42)*255
    plt.figure()
    plt.imshow(datacon_show, vmin = 0, vmax = 255)
    plt.figure()
    plt.imshow(data_show, vmin = 0, vmax = 255)
    
    # compare with test data
    datacon_show = data_con_test[figure_index_test,:].reshape(48,42)*255
    data_show = data_test[figure_index_test,:].reshape(48,42)*255
    plt.figure()
    plt.imshow(datacon_show, vmin = 0, vmax = 255)
    plt.figure()
    plt.imshow(data_show, vmin = 0, vmax = 255)
    
def result_mnist():
    num_train = 2000
    num_test = 3000 - num_train
    N = np.array([784,1000,500,250,2]) 
    data_train, label_train, data_test, label_test = mnist_read(num_train)
    
    # Read the data for weight matrix and bias
    # read the pretrained weights
    weight_vh1_final = np.loadtxt('./fine-tuning/20batch_pc/mnist/W_matrix_1_Final.txt')
    weight_vh2_final = np.loadtxt('./fine-tuning/20batch_pc/mnist/W_matrix_2_Final.txt')
    weight_vh3_final = np.loadtxt('./fine-tuning/20batch_pc/mnist/W_matrix_3_Final.txt')
    weight_vh4_final = np.loadtxt('./fine-tuning/20batch_pc/mnist/W_matrix_4_Final.txt')
    weight_vh5_final = np.loadtxt('./fine-tuning/20batch_pc/mnist/W_matrix_5_Final.txt')
    weight_vh6_final = np.loadtxt('./fine-tuning/20batch_pc/mnist/W_matrix_6_Final.txt')
    weight_vh7_final = np.loadtxt('./fine-tuning/20batch_pc/mnist/W_matrix_7_Final.txt')
    weight_vh8_final = np.loadtxt('./fine-tuning/20batch_pc/mnist/W_matrix_8_Final.txt')

    # construct the fine-tuning input weights
    w_list_4_layer_final = [weight_vh1_final, weight_vh2_final, weight_vh3_final, weight_vh4_final, weight_vh5_final, weight_vh6_final, weight_vh7_final, weight_vh8_final]

    # read the pretrained bias
    # hidden bias
    bias1_final = [np.loadtxt('./fine-tuning/20batch_pc/mnist/B_bias_1_Final.txt')]
    bias2_final = [np.loadtxt('./fine-tuning/20batch_pc/mnist/B_bias_2_Final.txt')]
    bias3_final = [np.loadtxt('./fine-tuning/20batch_pc/mnist/B_bias_3_Final.txt')]
    bias4_final = [np.loadtxt('./fine-tuning/20batch_pc/mnist/B_bias_4_Final.txt')]
    # visual bias
    bias5_final = [np.loadtxt('./fine-tuning/20batch_pc/mnist/B_bias_5_Final.txt')]
    bias6_final = [np.loadtxt('./fine-tuning/20batch_pc/mnist/B_bias_6_Final.txt')]
    bias7_final = [np.loadtxt('./fine-tuning/20batch_pc/mnist/B_bias_7_Final.txt')]
    bias8_final = [np.loadtxt('./fine-tuning/20batch_pc/mnist/B_bias_8_Final.txt')]
    # construct the fine-tuning input bias
    b_list_4_layer_final = [bias1_final, bias2_final, bias3_final, bias4_final, bias5_final, bias6_final, bias7_final, bias8_final]

    # for final
    hid1_train_final, hid2_train_final, hid3_train_final, hid4_train_final, vis5_train_final, vis6_train_final, vis7_train_final, data_con_train_final, map2d_train_final = autoencoder_output_conti_final(w_list_4_layer_final, b_list_4_layer_final, data_train, N, num_train)
    hid1_test_final, hid2_test_final, hid3_test_final, hid4_test_final, vis5_test_final, vis6_test_final, vis7_test_final, data_con_test_final, map2d_test_final = autoencoder_output_conti_final(w_list_4_layer_final, b_list_4_layer_final, data_test, N, num_test)

    #RBM result
    # read the pretrained weights
    weight_vh1 = np.loadtxt('./pretrain_20batch_pc_mnist/Weight_matrix_1.txt')
    weight_vh2 = np.loadtxt('./pretrain_20batch_pc_mnist/Weight_matrix_2.txt')
    weight_vh3 = np.loadtxt('./pretrain_20batch_pc_mnist/Weight_matrix_3.txt')
    weight_vh4 = np.loadtxt('./pretrain_20batch_pc_mnist/Weight_matrix_4.txt')
    # construct the fine-tuning input weights
    w_list_4_layer = [weight_vh1, weight_vh2, weight_vh3, weight_vh4]

    # read the pretrained bias
    # hidden bias
    bias1 = [np.loadtxt('./pretrain_20batch_pc_mnist/Hidden_bias_1.txt')]
    bias2 = [np.loadtxt('./pretrain_20batch_pc_mnist/Hidden_bias_2.txt')]
    bias3 = [np.loadtxt('./pretrain_20batch_pc_mnist/Hidden_bias_3.txt')]
    bias4 = [np.loadtxt('./pretrain_20batch_pc_mnist/Hidden_bias_4.txt')]
    # visual bias
    bias5 = [np.loadtxt('./pretrain_20batch_pc_mnist/Visible_bias_4.txt')]
    bias6 = [np.loadtxt('./pretrain_20batch_pc_mnist/Visible_bias_3.txt')]
    bias7 = [np.loadtxt('./pretrain_20batch_pc_mnist/Visible_bias_2.txt')]
    bias8 = [np.loadtxt('./pretrain_20batch_pc_mnist/Visible_bias_1.txt')]
    # construct the fine-tuning input bias
    b_list_4_layer = [bias1, bias2, bias3, bias4, bias5, bias6, bias7, bias8]
    
    # for pre-train
    hid1_train, hid2_train, hid3_train, hid4_train, vis5_train, vis6_train, vis7_train, data_con_train, map2d_train = autoencoder_output_conti(w_list_4_layer, b_list_4_layer, data_train, N, num_train)
    hid1_test, hid2_test, hid3_test, hid4_test, vis5_test, vis6_test, vis7_test, data_con_test, map2d_test = autoencoder_output_conti(w_list_4_layer, b_list_4_layer, data_test, N, num_test)
    
    # index number
    plt.figure()
    for i in np.arange(3):
        figure_index_train = np.rint(np.random.rand()*num_train)
        figure_index_test = np.rint(np.random.rand()*num_test)
    
    # compare with training data RBM & final
        datacon_show_train = data_con_train[figure_index_train,:].reshape(28,28)
        datacon_show_train_final = data_con_train_final[figure_index_train,:].reshape(28,28)
        data_show_train = data_train[figure_index_train,:].reshape(28,28)
        plt.subplot(3,3,1+i)
        plt.imshow(datacon_show_train)
        plt.axis('off')
        plt.subplot(3,3,2+i)
        plt.imshow(datacon_show_train_final)
        plt.axis('off')
        plt.subplot(3,3,3+i)
        plt.imshow(data_show_train)
        plt.axis('off')
    plt.show()
    
    # compare with test data RBM & final
    datacon_show_test = data_con_test[figure_index_test,:].reshape(28,28)
    datacon_show_test_final = data_con_test_final[figure_index_test,:].reshape(28,28)
    data_show_test = data_test[figure_index_test,:].reshape(28,28)
    plt.figure()
    plt.imshow(datacon_show_test)
    plt.axis('off')
    plt.figure()
    plt.imshow(datacon_show_test_final)
    plt.axis('off')
    plt.figure()
    plt.imshow(data_show_test)
    plt.axis('off')
    
    # draw 2-D map

    plt.figure()
    mymap = plt.cm.get_cmap('RdYlBu')
    plt.scatter(map2d_train[:,0],map2d_train[:,1],c=label_train[:,0],vmin=-1, vmax=1,cmap=mymap)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig('20batch_train_rbm.pdf',format='pdf')
    
    plt.figure()
    mymap = plt.cm.get_cmap('RdYlBu')
    plt.scatter(map2d_test[:,0],map2d_test[:,1],c=label_test[:,0],vmin=-1, vmax=1,cmap=mymap)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig('20batch_test_rbm.pdf',format='pdf')
    
    plt.figure()
    mymap = plt.cm.get_cmap('RdYlBu')
    plt.scatter(map2d_train_final[:,0],map2d_train_final[:,1],c=label_train[:,0],vmin=-1, vmax=1,cmap=mymap)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig('20batch_train_rbm_final.pdf',format='pdf')
    
    plt.figure()
    mymap = plt.cm.get_cmap('RdYlBu')
    plt.scatter(map2d_test_final[:,0],map2d_test_final[:,1],c=label_test[:,0],vmin=-1, vmax=1,cmap=mymap)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig('20batch_test_rbm_final.pdf',format='pdf')

    