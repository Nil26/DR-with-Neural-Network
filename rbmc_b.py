#!/usr/bin/python2.7
import numpy as np
import scipy
import numpy.random as random



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


# parameters of the data and output
numcases = 1000
numdim = 1000 # this is the number of visble dimensions
numbatches = 100
numhid = 20
#data
batchdata = np.zeros((numcases,numdim, numbatches))
# number of run
n = 1000


# calculate the probability of data and hidden for continues to binary RBM
def fun_prob_data(hid_data, Sigma, weight_vh, vibias):
    # here follows the gaussian function
	mean = np.add(np.multiply(np.dots(weight_vh, hid_data.transpose()).transpose(), sigma), np.repeat(vibias, numcases,axis=0)) 
    delta_mean = np.stract(data, mean)
    X = - 0.5 * np.divide(np.square(delta_mean), np.square(Sigma))
    coe = np.sqrt(1.0/(2 * np.pi))* np.reciprocal(np.sqrt(Sigma))
    prob = np.multiply(coe, np.exp(X))
    return prob



def fun_prob_hid(data, Sigma, weight_vh, hibias):
    Ones_m = np.full((numcases,numdim), 1.0)
    data_1 = np.divide(data, Sigma)
    #element inside logistic function
    X = -  np.dots(data_1, weight_vh) - np.repeat(hibias, numcases, axis=0)
    prob = np.divide(Ones_m, np.add(Ones_m, np.exp(X))    
    return prob



# function to use probability to decide the value of data or hidden variable
def fun_up(hid_data, data_hid_prob):
    hid_data = np.rint(data_hid_prob)# return to the nearest integer
    return hid_data

# function to use data and hidden to up data parameters
def fun_deltaweight(data, Sigma, hid_data, hid_con, data_con):
    weight_o = epsilonw /(numcases+0.0) * (np.dot(data.transpose(), hid_data) - np.dot(data_con.transpose(), hid_con))
    delta_weight =np.divide(weight_o, Sigma)
    return delta_weight

def fun_deltavbias(data, sigma, con_data, epsilonvb):
    return epsilonvb * np.divide((data.mean(0) - data_con.mean(0)), np.square(sigma))


def fun_deltahbias(hid_data, hid_con, epsilonhb):
    return epsilonhb * (hid_data.mean(0) - hid_con.mean(0))

def fun_deltasigma(data, vibias, hid_data, weight_vh, data_con, hid_con, Sigma, epsilonsig)
    three_m = np.full((numcases, numdim), 3)
    da_var = np.divide(np.square(data - np.repeat(vibias, numcases, axis=0), np.power(Sigma, three_m)))
    hid_var = np.(np.multiply(np.dot(weight_vh, hid_data.transpose()).transpose(), data), np.square(Sigma))
    orig_mean = np.mean((da_var-hid_var), 0)
    da_con_var = np.divide(np.square(data_con - np.repeat(vibias, numcases, axis=0), np.power(Sigma, three_m)))
    hid_con_var = np.(np.multiply(np.dot(weight_vh, hid_con.transpose()).transpose(), data_con), np.square(Sigma))
    con_mean = np.mean((da_con_var-hid_con_var), 0)
    return epsilonsig*(orig_mean - con_mean)

def fun_CD_k_c(k, data):
    data_k = np.zeros((k+1, numcases, numdim))
    data_k[0,:,:] = data
    hidden_k = np.zeros((k+1,numcases,numhid))
    #as sigma has different dimention to other matrixs
    for i in range(k):
        #construct hidden layer from data
        prob_hid = fun_prob_hid(data_k[k,:,:], Sigma, weight_vh, hibias)
        hidden_k[k] = fun_up(hidden_k[k],prob_hid)
        #construct confabulation state from hidden layers
        prob_data = fun_prob_data(hid_data, Sigma, weight_vh, vibias)
        data_k[k+1] = fun_up(data_k[k+1],con_dataprob)
    #construct confabulation hidden state from confabulation data
    hidprob = fun_prob_hid(data_k[k+1], weight_vh, hibias)
    hid_con = fun_uphid(hid_con, hidprob)
    return hid_con, data_k[k+1]

##############################################################################
#define continuous to binary PBM
#####################################################################

def fun_RBM_con(bachdata, numdim, numhid, numcases):
    #initializeing symmetric weights and bias
    weight_vh = 0.1 * random.rand(numdim, numhid)
    hibias = np.zeros((1,numhid))
    vibias = np.zeros((1,numdim))
    sigma = np.var(data, 0).reshape((1, numdim))

    # probabilities we may need while calculation
    data_hidprob = np.zeros((numcases, numhid))
    all_data_hidprob = np.zeros((numdim, numhid))
    con_data_hidprob = np.zeros((numcases, numhid))
    all_con_data_hidprob = np.zeros((numdim, numhid))
    con_hid_dataprob = np.zeros(numcases, numdim)
    # change of parameters
    delta_weight = np.zeros((numdim, numhid))
    delta_vibias = np.zeros((1,numdim))
    delta_hibias = np.zeros((1,numhid))
    delta_sigma = np.zeros((1,numdim))
    
    #out put data
    hid_data = np.zeros((numcases, numhid))
    hid_con = np.zeros((numcases, numhid))
    data_con = np.zeros((numcases, numhid))

    #loop
    for iteration in range(n):
        for batch in range(numbatches):
            data = batchdata[:,:,batch]
            #CD-k algorithm
            Sigma = np.repeat(sigma, numcasesm, axis=0)
            hid_con, data_con = fun_CD_k_c(k, data)
            # update parameters:
            delta_weight = fun_deltaweight(data, Sigma, hid_data, hid_con, data_con)
            delta_hibias = fun_deltahbias(hid_data, hid_con, epsilonhb)
            delta_vibias = fun_deltavbias(data, sigma, con_data, epsilonvb)
            delta_sigma = fun_deltasigma(data, vibias, hid_data, weight_vh, data_con, hid_con, Sigma, epsilonsig)
            weight_vh = weight_vh + delta_weight
            hibias = hibias + delta_hibias
            vibias = vibias + delta_vibias
            if delta_weight < 0.001 and hibias < 0.001 and vibias < 0.001:
                break
    prob = fun_prob(data, weight_vh, hibias)
    hid_data = fun_uphid(hid_data, prob)
    return hid_data, weight_vh, hibias, vibias