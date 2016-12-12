import numpy as np 
import scipy.io as sio 
from numpy import linalg as LA
import matplotlib.pyplot as plt 
import time
yale = sio.loadmat('yalefaces.mat')
yalefaces = yale['yalefaces']
x_data = yalefaces.reshape(2016,2414)
x_mean = np.mean(x_data,axis=1)
# first we need to calculate the matrix S
S = np.zeros((2016,2016))
for i in range(2414):
    S += 1.0/2414*np.outer(x_data[:,i]-x_mean,(x_data[:,i]-x_mean).transpose())
w, v = LA.eig(S)
w_1 = np.sort(w)
w_index = np.argsort(w)
w_s = np.zeros((2016,))
w_index_s = np.zeros((2016,))
for i in range(2016):
    w_s[i] = w_1[2015-i]
    w_index_s[i] = w_index[2015-i]
#solve for u
u, Lambda, u_1 = np.linalg.svd(S)
t = np.linspace(0,2016,2016)
plt.semilogy(t, w_s)
plt.grid(True)
plt.show()

#decide the number of the componenets we need
n = 0
var = 0
var_tot = 2016 * np.mean(w_s)
for i in range(2016):
    lambda_i = w_s[i]
    n += 1
    var += (lambda_i+0.0)/var_tot
    if var >= 0.99:
      break


for i in range(20):
    if i == 0:
        x = np.mean(yalefaces,axis=2)
    else:
        j = w_index_s[i-1]
        x = v[:,j].reshape((48,42))
    ax = plt.subplot(4, 5, i+1)
    plt.imshow(x, extent=[0, 1, 0, 1]) 
    plt.imshow(x, cmap=plt.get_cmap('gray')) 
#    time.sleep(0.1) 
plt.show()