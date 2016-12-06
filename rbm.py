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




# parameters of the data and output
numcases = 1000
numdim = 1000 # this is the number of visble dimensions
numbatches = 100
numhid = 20
#data
batchdata = np.zeros((numcases,numdim, numbatches))
# number of run
n = 1000


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


# out put data
hid_data = np.zeros((numcases, numhid))
hid_con = np.zeros((numcases, numhid))
data_con = np.zeros((numcases, numhid))



def func_prob(data_hidprob, data, weight_vh, hibias):
    for i in range(numcases):    
        for j in range(numhid):
            data_hidprob[i][j] = 1./(1+np.exp(-np.dot(data[i,:], weight_vh[:,j])-hibias[0,j]))
    return data_hidprob

def fun_uphid(hid_data, data_hid_prob):
    for i in range(numcases):
        for j in range(numhid):
            if data_hid_prob >= 0.5:
                hid_data[i,j] = 1
            else:
                hid_data[i,j] = 0
    return hid_data

def fun_delta_weight(data, hid_data, hid_con, data_con):
    return epsilonw * (np.dot(data.transpose())

for iteration in range(n):
    for batch in range(numbatches):
        data = batchdata[:,:,batch]
        #construct hidden layer from data
        data_hidprob = func_prob(data_hidprob, data, weight_vh, hibias)
        hid_data = fun_uphid(hid_data,data_hidprob)
        #construct confabulation state from hidden layers
        con_hid_dataprob = func_prob(con_hid_dataprob, hid_data, weight_vh, vibias)
        data_con = fun_uphid(data_con,con_hid_dataprob)
        #construct confabulation hidden state from confabulation data
        con_data_hidprob = func_prob(con_data_hidprob, data_con, weight_vh, hibias)
        hid_con = fun_uphid(hid_con, con_data_hidprob)
        # update parameters:

        
        
        
        
