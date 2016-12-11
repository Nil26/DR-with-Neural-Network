#!/usr/bin/python2.7
import numpy as np
import numpy.random as random



# parameters of RBM
epsilonw = 0.1 #learning rate for weights
epsilonvb = 0.1 #learning rate for bias for visible units
epsilonhb = 0.1 # learning rate for bias for hidden units

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




# calculate the probability of data or hidden variable for normal binary to binary RBM
def fun_prob_h(data, weight_vh, hibias):
    Ones_m = np.full((numcases,numhid), 1.0)
    #element inside logistic function
    X = - np.dots(data, weight_vh) - np.repeat(hibias, axis=0)
    prob = np.divide(Ones_m, np.add(Ones_m, np.exp(X)))    
    return prob

def fun_prob_d(hid_data, weight_vh, vibias):
    Ones_m = np.full((numcases,numdim), 1.0)
    #element inside logistic function
    X = - np.dots(weight_vh, hid_data.transpose()).transpose() - np.repeat(vibias, numcases, axis=0)
    prob = np.divide(Ones_m, np.add(Ones_m, np.exp(X)))    
    return prob



# function to use probability to decide the value of data or hidden variable
def fun_uphid(data_hid_prob):
    hid_data = np.rint(data_hid_prob)# return to the nearest integer
    return hid_data

# function to use data and hidden to up data parameters
def fun_delta_weight(data, hid_data, hid_con, data_con):
    delta_weight =epsilonw /(numcases+0.0) * (np.dot(data.transpose(), hid_data) - np.dot(data_con.transpose(), hid_con))
    return delta_weight

def fun_delta_bias(data, con_data, epsilonvb):
    return epsilonvb * (np.mean(data, axis=0) - np.mean(con_data, axis=0))

def energy_cal(data, weight_vh, vibias, hibias):
    hid_prob = fun_prob_h(data, weight_vh, hibias)
    hid = fun_uphid(hid_prob)
    energy =  np.mean(np.dot(vibias, data.transpose())) +  np.mean(np.dot(hibias, hid.transpose())) + np.mean(np.multiply(np.dot(data, weight_vh), hid))
    return energy

def fun_CD_k(k, data, weight_vh, hibias, vibias):
    # probabilities we may need while calculation
    prob_hid = np.zeros((numcases, numhid))
    hidprob = np.zeros((numcases, numhid))
    prob_data = np.zeros(numcases, numdim)    
    
    data_k = np.zeros((k+1, numcases, numdim))
    data_k[0,:,:] = data
    hidden_k = np.zeros((k+1,numcases,numhid))
    for i in range(k):
        #construct hidden layer from data
        prob_hid = fun_prob_h(data_k[k,:,:], weight_vh, hibias)
        hidden_k[k] = fun_uphid(prob_hid)
        #construct confabulation state from hidden layers
        prob_data = fun_prob_d(weight_vh, vibias)
        data_k[k+1] = fun_uphid(prob_data)
    #construct confabulation hidden state from confabulation data
    hidprob = fun_prob_h(data_k[k+1], weight_vh, hibias)
    hid_con = fun_uphid(hidprob)
    return hidden_k[0], hid_con, data_k[k+1]


#####################################################################################################
#define one layer RBM (normal)
#################################################################################
def fun_RBM(batchdata, numhid):
    # parameters of the data and output
    numcases, numdim, numbatches = batchdata.shape
    #number of data
    N_t = numcases * numbatches    
    #initializeing symmetric weights and bias
    weight_vh = 0.1 * random.rand(numdim, numhid)
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
    data_1 = batchdata.reshape(N_t, numdim)
    for iteration in range(n):
        E_fun[iteration] = energy_cal(data_1, weight_vh, vibias, hibias)
        for batch in range(numbatches):
            data = batchdata[:,:,batch]
            #CD-k algorithm
            hid_data, hid_con, data_con = fun_CD_k(k, data)
            # update parameters:
            delta_weight = fun_delta_weight(data, hid_data, hid_con, data_con)
            delta_hibias = fun_delta_bias(hid_data, hid_con, epsilonhb)
            delta_vibias = fun_delta_bias(data, data_con, epsilonvb)
            weight_vh = weight_vh + delta_weight
            hibias = hibias + delta_hibias
            vibias = vibias + delta_vibias
            if delta_weight < 0.001 and hibias < 0.001 and vibias < 0.001:
                break
        if delta_weight < 0.001 and hibias < 0.001 and vibias < 0.001:
            E_fun[iteration+1] = energy_cal(data_1, weight_vh, vibias, hibias)
            break

    return hid_data, weight_vh, hibias, vibias, E_fun
        
