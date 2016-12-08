#!/usr/bin/python2.7
import numpy as np
import scipy
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
numdim = 1000 # this is the number of visible dimensions
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
<<<<<<< HEAD
<<<<<<< HEAD
    X = - np.dots(data, weight_vh) - np.repeat(hibias, axis=0)
    prob = np.divide(Ones_m, np.add(Ones_m, np.exp(X))    
    return prob

def fun_prob_d(hid_data, weight_vh, vibias):
    Ones_m = np.full((numcases,numdim), 1.0)
    #element inside logistic function
    X = - np.dots(weight_vh, hid_data.transpose()).transpose() - np.repeat(vibias, numcases, axis=0)
    prob = np.divide(Ones_m, np.add(Ones_m, np.exp(X))    
=======
    X = - np.dots(data, weight_vh) - hibias
    prob = np.divide(Ones_m, np.add(Ones_m, np.exp(X)))
>>>>>>> origin/master
=======
    X = - np.dots(data, weight_vh) - hibias
    prob = np.divide(Ones_m, np.add(Ones_m, np.exp(X)))
>>>>>>> d0e63d30bd61dfed2dbda829329586d41741aea0
    return prob



# function to use probability to decide the value of data or hidden variable
def fun_uphid(hid_data, data_hid_prob):
    hid_data = np.rint(data_hid_prob)# return to the nearest integer
    return hid_data

# function to use data and hidden to up data parameters
def fun_delta_weight(data, hid_data, hid_con, data_con):
    delta_weight =epsilonw /(numcases+0.0) * (np.dot(data.transpose(), hid_data) - np.dot(data_con.transpose(), hid_con))
    return delta_weight

def fun_delta_bias(data, con_data, epsilonvb):
    return epsilonvb * (data.mean(1) - data_con.mean(2))


def fun_CD_k(k, data):
    data_k = np.zeros((k+1, numcases, numdim))
    data_k[0,:,:] = data
    hidden_k = np.zeros((k+1,numcases,numhid))
    for i in range(k):
        #construct hidden layer from data
        prob_hid = fun_prob_h(data_k[k,:,:], weight_vh, hibias)
        hidden_k[k] = fun_uphid(hidden_k[k],prob_hid)
        #construct confabulation state from hidden layers
        prob_data = fun_prob_d(hid_data, weight_vh, vibias)
        data_k[k+1] = fun_uphid(data_k[k+1],con_dataprob)
    #construct confabulation hidden state from confabulation data
    hidprob = fun_prob_h(data_k[k+1], weight_vh, hibias)
    hid_con = fun_uphid(hid_con, hidprob)
    return hidden_k[0], hid_con, data_k[k+1]


#####################################################################################################
#define one layer RBM (normal)
#################################################################################
def fun_RBM(batchdata, numdim, numhid, numcases)
    #initializeing symmetric weights and bias
    weight_vh = 0.1 * random.rand(numdim, numhid)
    hibias = np.zeros((1,numhid))
    vibias = np.zeros((1,numdim))

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
    
    
    #out put data
    hid_data = np.zeros((numcases, numhid))
    hid_con = np.zeros((numcases, numhid))
    data_con = np.zeros((numcases, numhid))
    #loop
    for iteration in range(n):
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
    prob = fun_prob(data, weight_vh, hibias)
    hid_data = fun_uphid(hid_data, prob)
    return hid_data, weight_vh, hibias, vibias
        
