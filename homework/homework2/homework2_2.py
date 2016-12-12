#!/user/bin/python
import numpy as np
import scipy.io as sio
from numpy import linalg as LA
from numpy.linalg import inv
import matplotlib.pyplot as plt

#since we have to use Newton's methods, in this particlar question we have
# objetive function J = -l(\theta) + \lambda \|\|\theta\|\|^2
#Then Newton-Raphson will become
#\theta_{t+1} = \theta_{t} -(\nabla^{2}J(\theta_{t}))^{-1}\nabla J(\theta_t)
# then we define the first derivative and second derivative function as well as objective function
def OFCT(x_list, y_list, theta_list):
    sum = 0
    for i in range(2000):
        d = y_list[0,i]
        A = np.exp(-y_list[0,i] * np.inner(theta_list.transpose(), x_list[:,i]))
        sum += np.log(1 + A)
    return sum + 2 * 10 * LA.norm(theta_list)**2 

def FirstD_OFCT(x_list, y_list, theta_list):
    sum = np.zeros((785))
    for i in range(2000):
        d = y_list[0,i]
        A = np.exp(-y_list[0,i] * np.inner(theta_list.transpose(), x_list[:,i]))
        sum += A /(1 + A) * (-d) * x_list[:,i]
    return sum + 2*10*theta_list
    
def SecondD_OFCT(x_list, y_list, theta_list):
    sum_2 = np.zeros((785,785))
    for i in range(2000):
        d = y_list[0,i]
        A = np.exp(-d * (np.dot(theta_list.transpose(),x_list[:,i])))
        sum_2 += A / (1+A)**2 * d**2 * np.outer(x_list[:,i].transpose(), x_list[:,i])
    return sum_2 + 2 * 10 * np.identity(785)

#import data 
mnist_49_3000 = sio.loadmat('C:/Users/zheng/OneDrive/Documents/Physics/Machine learning/home work/mnist_49_3000.mat')
x = mnist_49_3000['x']
d,n = x.shape
x_1 = np.insert(x, 0, 1, axis=0)
y = mnist_49_3000['y']
#choose initialized \theta vector to start
theta_list = np.zeros((785,10000))
theta_fit = np.zeros((785))

#here is to find the theta minizing the Objetive function using the trainning data
for j in range(100000):
    theta_list[:,j+1] = theta_list[:,j] - np.inner(inv(SecondD_OFCT(x_1, y, theta_list[:,j])), FirstD_OFCT(x_1, y, theta_list[:,j]))
    theta_fit = theta_list[:,j+1]
    if LA.norm(theta_list[:,j+1] - theta_list[:,j])  < 0.0001:
        break
#then we have the theta_fit as the optimal value, now we can calculate the objective function for theta_fit

L = OFCT(x_1,y, theta_fit)
print L

#then use the theta we find to classify the imput data, we have the classify function
# that is if theta^T * x_i > 0 then y_i should be 1, others should be -1
y_CLASS = np.zeros(1000)
N_diff = 0
Error_list = np.zeros((1000)) # creat a list to restore the error events
for k in range(2000, 3000):
    if np.inner(theta_fit.transpose(),x_1[:,k]) >= 0:
        y_CLASS[k-2000] = 1
    else:
        y_CLASS[k-2000] = -1
    if y_CLASS[k-2000] != y[0,k]:
        N_diff += 1
        Error_list[k-2000] = -np.abs(np.inner(theta_fit.transpose(),x_1[:,k]))
#        print k
#        print np.inner(theta_fit.transpose(),x_1[:,k])
#       print y[0,k]
print N_diff

#to find the 20 gaphes

Error_list_sort = np.argsort(Error_list)
for l in range(20):
    a = l / 4
    b = l % 4
    m = Error_list_sort[l]
    ax = plt.subplot(4, 5, l+1)
    plt.imshow( np.reshape(x[:,m], (int(np.sqrt(d)), int (np.sqrt(d)))))
    plt.title(y[0,m])
#    plt.Axes.get_xaxis(False)
#    plt.Axes.get_yaxis(False)
    plt.tight_layout()

    


    
    