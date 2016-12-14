#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 15:43:51 2016

@author: zhshang
"""

import numpy as np
import scipy.io as sio

# for fine-tuning
from fine_tuning import fine_tuning

def face_read(n_train):
    yale = sio.loadmat('yalefaces.mat')
    yalefaces = yale['yalefaces']
    # normalize the color data
    face_data = yalefaces.reshape(2016,2414)
    face_norm = face_data/255
    data = np.transpose(face_norm)
    data_train = data[0:n_train,:]
    data_test = data[n_train:,:]
    return data_train, data_test

# For doing the fine-tuning process
# constants used in fine-tuning
training_steps = 200
num_batch = 100
lr = 0.01

# read the pretrained weights
weight_vh1 = np.loadtxt('/home/zhshang/DeepEncoder/yale_faces/Weight_matrix_1.txt')
weight_vh2 = np.loadtxt('/home/zhshang/DeepEncoder/yale_faces/Weight_matrix_2.txt')
weight_vh3 = np.loadtxt('/home/zhshang/DeepEncoder/yale_faces/Weight_matrix_3.txt')
weight_vh4 = np.loadtxt('/home/zhshang/DeepEncoder/yale_faces/Weight_matrix_4.txt')
# construct the fine-tuning input weights
w_list_4_layer = [weight_vh1, weight_vh2, weight_vh3, weight_vh4]

# read the pretrained bias
# hidden bias
bias1 = np.array([np.loadtxt('/home/zhshang/DeepEncoder/yale_faces/Hidden_bias_1.txt')]).transpose()
bias2 = np.array([np.loadtxt('/home/zhshang/DeepEncoder/yale_faces/Hidden_bias_2.txt')]).transpose()
bias3 = np.array([np.loadtxt('/home/zhshang/DeepEncoder/yale_faces/Hidden_bias_3.txt')]).transpose()
bias4 = np.array([np.loadtxt('/home/zhshang/DeepEncoder/yale_faces/Hidden_bias_4.txt')]).transpose()
# visual bias
bias5 = np.array([np.loadtxt('/home/zhshang/DeepEncoder/yale_faces/Visible_bias_4.txt')]).transpose()
bias6 = np.array([np.loadtxt('/home/zhshang/DeepEncoder/yale_faces/Visible_bias_3.txt')]).transpose()
bias7 = np.array([np.loadtxt('/home/zhshang/DeepEncoder/yale_faces/Visible_bias_2.txt')]).transpose()
bias8 = np.array([np.loadtxt('/home/zhshang/DeepEncoder/yale_faces/Visible_bias_1.txt')]).transpose()
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

