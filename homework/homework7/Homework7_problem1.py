from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
#To display this image 
im = Image.open('mandrill.tiff') 
#im.show()
y = np.array(im) 
M = 2 #block side-length 
n = np.prod(y.shape)/(3*M*M) #number of blocks 
d = y.shape[0] 
c = 0 # counter 
x = np.zeros([n,3*M*M]) 
for i in range(0,d,M): 
    for j in range(0,d,M):
      #   print c,i,j,M,y[i:i+M,j:j+M,:].shape, M*M*3 
         x[c,:] = np.reshape(y[i:i+M,j:j+M,:], [1,M*M*3]) 
         c = c+1
#Test
k = 100

#dist
def dist2(x,c):
    ndata,dimx = x.shape
    ncenters, dimc = c.shape

    xsum = np.sum(x**2,axis = 1)
    xsum = xsum[:,np.newaxis]
    csum = np.sum(c**2,axis = 1)
    csum = csum[:,np.newaxis]

    n2 =  xsum.dot(np.ones([1,ncenters]))+ np.ones([ndata,1]).dot(csum.T)- 2*x.dot(c.T)
    return n2

#Definition for Objective function
def OBF(C_n):
    TOT = 0.0
    for i in range(k):
          TOT += 1.0/(np.where(C_n == i)[0].size)*np.sum(dist2(x[np.where(C_n == i)[0],:],x[np.where(C_n == i)[0],:]))
    return 0.5 * TOT
#initialization
np.random.seed(0) 
perm = np.random.permutation(n) 
m = x[perm[0:k],:] # initial cluster centers
Mean = np.zeros((10000,100,12))
C_n = np.zeros((10000,n))
OBT = np.zeros((10000,))
# have the mean for each block
for l in range(10000):
    if l == 0:
        C_nk = dist2(x,m)
    else:
        C_nk = dist2(x,Mean[l-1,:,:])
    C_n[l,:] = np.argmin(C_nk,axis=1)# to find the k center
    for j in range(k):
        Mean[l,j,:] = np.mean(x[np.where(C_n[l,:]==j)[0],:],axis=0)
    OBT[l] = OBF(C_n[l,:])
    if l  > 0:
        if (Mean[l,:,:]==Mean[l-1,:,:]).all() and (C_n[l,:] == C_n[l-1,:]).all():
            break
# Reconstruct Replace
x_r = np.zeros((n,12))
for i in range(n):
    index = C_n[120,i]
    x_r[i,:] = m[index]

d = y.shape[0] 
c = 0
y_r = np.zeros(y.shape)
M = 2
for i in range(0,d,M): 
    for j in range(0,d,M):
      #   print c,i,j,M,y[i:i+M,j:j+M,:].shape, M*M*3 
      y_r[i:i+M,j:j+M,:] = np.reshape(x_r[c,:], [M,M,3]) 
      c += 1
y_1 = y.astype(np.float32) 
plt.imshow(y_1/256)    
plt.imshow(y_r/256)
plt.imshow((y_1-y_r)/256)
plt.colorbar()
iteration = np.linspace(0,l,l)
plt.plot(iteration,OBT[:l])

plt.show()



    