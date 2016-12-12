#!/usr/bin/python2.7
import numpy as np
import numpy.random as random
from numpy import linalg as LA
import time


# parameters of RBM
epsilonw = 0.1 #learning rate for weights
epsilonvb = 0.1 #learning rate for bias for visible units
epsilonhb = 0.1 # learning rate for bias for hidden units
epsilonsig = 0.1
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
##number of data
#N_t = numcases * numbatches
# number of run
n = 5000

# calculate the probability of data and hidden for continues to binary RBM
# calculation for update data and hidden variables
def fun_data_cal(hid_data, weight_vh, vibias, numcases):
    # here follows the gaussian function
    mean = np.add(np.dot(weight_vh, hid_data.transpose()).transpose(), np.repeat(vibias, numcases,axis=0)) 
    return mean



def fun_prob_hid(data, weight_vh, hibias, numcases, numhid):
    Ones_m = np.full((numcases,numhid), 1.0)
    #element inside logistic function
    #print((np.repeat(hibias, numcases, axis=0)).shape)
    X = -  np.dot(data, weight_vh) - np.repeat(hibias, numcases, axis=0)
    prob = np.divide(Ones_m, np.add(Ones_m, np.exp(X)))
    return prob



# function to use probability to decide the value of data or hidden variable
def fun_up(data_hid_prob):
    hid_data = np.rint(data_hid_prob)# return to the nearest integer
    return hid_data

# function to use data and hidden to up data parameters
def fun_deltaweight(data, hid_data, hid_con, data_con, numcases):
    weight_o = epsilonw /(numcases+0.0) * (np.dot(data.transpose(), hid_data) - np.dot(data_con.transpose(), hid_con))
    return weight_o

def fun_deltavbias(data, con_data, epsilonvb):
    return epsilonvb * (np.mean(data, axis=0) - np.mean(con_data,axis=0))


def fun_deltahbias(hid_data, hid_con, epsilonhb):
    return epsilonhb * (np.mean(hid_data, axis=0) - np.mean(hid_con,axis=0))

def energy_cal(data, weight_vh, vibias, hibias, number_tot, numhid):
    hid_prob = fun_prob_hid(data, weight_vh, hibias, number_tot, numhid)
    hid = fun_up(hid_prob)
    energy =  0.5 * np.mean(np.square(np.multiply(data-np.repeat(vibias, number_tot, axis=0))), axis = 0) +  np.mean(np.dot(hibias, hid.transpose())) + np.mean(np.multiply(np.dot(data, weight_vh), hid))
    return -energy    

def fun_CD_k_c(k, data, weight_vh, vibias, hibias, numcases, numdim ,numhid):
        # probabilities we may need while calculation
    prob_hid = np.zeros((numcases, numhid))
    hidprob = np.zeros((numcases, numhid))
    prob_data = np.zeros((numcases, numdim))
    data_k = np.zeros((k+1, numcases, numdim))
    data_k[0,:,:] = data
    hidden_k = np.zeros((k+1,numcases,numhid))
    #as sigma has different dimention to other matrixs
    for i in range(k):
        #construct hidden layer from data
        prob_hid = fun_prob_hid(data_k[i,:,:], weight_vh, hibias, numcases, numhid)
        hidden_k[i] = fun_up(prob_hid)
        #construct confabulation state from hidden layers
        prob_data = fun_data_cal(hidden_k[i], weight_vh, vibias, numcases)
        data_k[i+1] = prob_data
    #construct confabulation hidden state from confabulation data
    hidprob = fun_prob_hid(data_k[k], weight_vh, hibias, numcases, numhid)
    hid_con = fun_up(hidprob)
    return hid_con, data_k[k]

##############################################################################
#define continuous to binary PBM
#####################################################################

def fun_RBM_con(batchdata, numhid):
    # parameters of the data and output
    numcases, numdim, numbatches = batchdata.shape
    #number of data
    N_t = numcases * numbatches    

    #initializeing symmetric weights and bias
    weight_vh = random.rand(numdim, numhid)
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

    # energy calculation
    E_fun = np.zeros((n,))
    #loop
    data_1 = batchdata.reshape(N_t, numdim)
    for iteration in range(n):
        E_fun[iteration] = energy_cal(data_1, weight_vh, vibias, hibias, N_t, numhid)
        for batch in range(numbatches):
            data = batchdata[:,:,batch]
            #CD-k algorithm
            hid_con, data_con = fun_CD_k_c(k, data, weight_vh, vibias, hibias, numcases, numdim ,numhid)
            # update parameters:
            delta_weight = fun_deltaweight(data, hid_data, hid_con, data_con, numcases)
            delta_hibias = fun_deltahbias(hid_data, hid_con, epsilonhb)
            delta_vibias = fun_deltavbias(data, data_con, epsilonvb)
            weight_vh = weight_vh + delta_weight
            hibias = hibias + delta_hibias
            vibias = vibias + delta_vibias
            if LA.norm(delta_weight) < 0.1 and LA.norm(delta_hibias) < 0.1 and LA.norm(delta_vibias) < 0.1:
                break
        if LA.norm(delta_weight) < 0.1 and LA.norm(delta_hibias) < 0.1 and LA.norm(delta_vibias) < 0.1:
            E_fun[iteration+1] = energy_cal(data_1, weight_vh, vibias, hibias, N_t, numhid)      
            break
    prob = fun_prob_hid(data_1, weight_vh, hibias, N_t, numhid)
    hid_data_1 = fun_up(prob)
    hid_data = hid_data_1.reshape(numcases, numhid, numbatches)
    return hid_data, weight_vh, hibias, vibias, E_fun