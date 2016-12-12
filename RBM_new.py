# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 11:11:03 2016

@author: zheng
"""

#!/usr/bin/python2.7
import numpy as np
import numpy.random as random
from numpy import linalg as LA


# parameters of RBM
epsilonw = 0.1 #learning rate for weights
epsilonvb = 0.1 #learning rate for bias for visible units
epsilonhb = 0.1 # learning rate for bias for hidden units

weightcost = 0.0002
initialmomentum = 0.5
finalmomentum = 0.9

#parameter to decide algorithm
k = 1

## parameters of the data and output
#numcases = 1000
#numdim = 1000 # this is the number of visble dimensions
#numbatches = 100
#numhid = 20
##data
#batchdata = np.zeros((numcases,numdim, numbatches))

# number of run
n = 500




# calculate the probability of data or hidden variable for normal binary to binary RBM
def fun_prob_h(data, weight_vh, hibias, numcases, numhid):
    Ones_m = np.full((numcases,numhid), 1.0)
    #element inside logistic function
    X = - np.dot(data, weight_vh) - np.repeat(hibias, numcases, axis=0)
    prob = np.divide(Ones_m, np.add(Ones_m, np.exp(X)))    
    return prob

def fun_prob_d(hid_data, weight_vh, vibias, numcases, numdim):
    Ones_m = np.full((numcases,numdim), 1.0)
    #element inside logistic function
    X = - np.dot(weight_vh, hid_data.transpose()).transpose() - np.repeat(vibias, numcases, axis=0)
    prob = np.divide(Ones_m, np.add(Ones_m, np.exp(X)))    
    return prob



# function to use probability to decide the value of data or hidden variable
def fun_uphid(data_hid_prob):
    hid_data = np.rint(data_hid_prob)# return to the nearest integer
    return hid_data

# function to use data and hidden to up data parameters
def fun_delta_weight(data, prob_hid, hidprob, data_con, numcases):
    delta_weight =epsilonw /(numcases+0.0) * (np.dot(data.transpose(), prob_hid) - np.dot(data_con.transpose(), hidprob))
    return delta_weight

def fun_delta_bias(data, con_data, epsilonvb):
    return epsilonvb * (np.mean(data, axis=0) - np.mean(con_data, axis=0))

def energy_cal(data, weight_vh, vibias, hibias, number_tot, numhid):
    hid_prob = fun_prob_h(data, weight_vh, hibias, number_tot, numhid)
    hid = fun_uphid(hid_prob)
    energy =  np.mean(np.dot(vibias, data.transpose())) +  np.mean(np.dot(hibias, hid.transpose())) + np.mean(np.multiply(np.dot(data, weight_vh), hid))
    return -energy

def fun_CD_k(k, data, weight_vh, hibias, vibias, numcases, numdim, numhid):
    # probabilities we may need while calculation
    prob_hid = np.zeros((numcases, numhid))
    hidprob = np.zeros((numcases, numhid))
    prob_data = np.zeros((numcases, numdim))    
    
    data_k = np.zeros((k+1, numcases, numdim))
    data_k[0,:,:] = data
    hidden_k = np.zeros((k+1,numcases,numhid))
    for i in range(k):
        #construct hidden layer from data
        prob_hid = fun_prob_h(data_k[i,:,:], weight_vh, hibias, numcases, numhid)
        hidden_k[i] = fun_uphid(prob_hid)
        #construct confabulation state from hidden layers
        prob_data = fun_prob_d(hidden_k[i], weight_vh, vibias, numcases, numdim)
        data_k[i+1] = prob_data
    #construct confabulation hidden state from confabulation data
    hidprob = fun_prob_h(data_k[k], weight_vh, hibias, numcases, numhid)
    hid_con = fun_uphid(hidprob)
    return hidden_k[0], prob_hid, hid_con, data_k[k], hidprob


#####################################################################################################
#define one layer RBM (normal)
#################################################################################
def fun_RBM_new(batchdata, numhid):
    # parameters of the data and output
    numcases, numdim, numbatches = batchdata.shape
    #number of data
    N_t = numcases * numbatches    
    #initializeing symmetric weights and bias
    weight_vh = 0.2 * (random.rand(numdim, numhid)-0.5)
    hibias = np.zeros((1,numhid))
    vibias = np.zeros((1,numdim))


    # change of parameters
    delta_weight = np.zeros((numdim, numhid))
    delta_vibias = np.zeros((1,numdim))
    delta_hibias = np.zeros((1,numhid))
    
    
    #out put data
    hid_data = np.zeros((numcases, numhid))
    hid_con = np.zeros((numcases, numhid))
    data_con = np.zeros((numcases, numhid))
    #loop
    E_fun = np.zeros((n,))
    err = np.zeros((n,))
    data_1 = np.transpose(batchdata,(0,2,1)).reshape(N_t, numdim)
    for iteration in range(n):
        E_fun[iteration] = energy_cal(data_1, weight_vh, vibias, hibias, N_t, numhid)
        for batch in range(numbatches):
            data = batchdata[:,:,batch]
            #CD-k algorithm
            hid_data, prob_hid, hid_con, data_con, hidprob = fun_CD_k(k, data, weight_vh, hibias, vibias, numcases, numdim, numhid)
            # update parameters:
            delta_weight = fun_delta_weight(data, prob_hid, hidprob, data_con, numcases)
            delta_hibias = fun_delta_bias(hid_data, hid_con, epsilonhb)
            delta_vibias = fun_delta_bias(data, data_con, epsilonvb)
            weight_vh = weight_vh + delta_weight
            hibias = hibias + delta_hibias
            vibias = vibias + delta_vibias
        data_2 = fun_prob_h(data_1, weight_vh, hibias, N_t, numhid)
        data_3 = fun_prob_d(data_2, weight_vh, vibias, N_t, numdim)
        err[iteration] = LA.norm(data_1 - data_3)
        if iteration > 1:
            if err[iteration]/N_t < 0.1 or (LA.norm(delta_weight) < 0.1 and LA.norm(delta_hibias) < 0.1 and LA.norm(delta_vibias) < 0.1):
                E_fun[iteration+1] = energy_cal(data_1, weight_vh, vibias, hibias, N_t, numhid)
                break
    prob = fun_prob_h(data_1, weight_vh, hibias, N_t, numhid)
    hid_data_1 = fun_uphid(prob)
    hid_data = np.transpose(hid_data_1.reshape(numcases, numbatches, numhid), (0,2,1))
    return hid_data, weight_vh, hibias, vibias, E_fun, err
        
