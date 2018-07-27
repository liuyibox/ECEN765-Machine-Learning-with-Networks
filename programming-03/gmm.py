import numpy as np

path_train = 'bclass-train'
path_test  = 'bclass-test'
data_train = []
data_test = []

try:
    with open(path_train) as f:
        for line in f:
            data_train.append([float(number) for number in line.split()])
    with open(path_test) as f:
        for line in f:
            data_test.append([float(number) for number in line.split()])

except IOError:
    print("error when reading train data\n")

data_train = np.array(data_train)
#label_train = data_train[:,0]
#label_train = label_train.astype(np.int64)
#bias_train = np.ones((len(data_train),1))
data_train = data_train[:,1:35]
data_train = np.c_[data_train[:,0],data_train[:,2:34]]

data_test = np.array(data_test)
#label_test = data_test[:,0]
#label_test = label_test.astype(np.int64)
#bias_test = np.ones((len(data_test),1))
data_test = data_test[:,1:35]
data_test = np.c_[data_test[:,0],data_test[:,2:34]]

k = 10
m = np.random.rand(k, data_train[0].size)
s = np.zeros((k, data_train[0].size, data_train[0].size),dtype=float)
for i in range(k):
    temp_s = np.random.rand(data_train[0].size, data_train[0].size)
    s[i] = np.dot(temp_s, temp_s.transpose())
    if((s[i].transpose() == s[i]).all()):
        print "we have a symmetric matrix"
    else:
        print "we have a non-symmetric matrix"

pi = np.array([float(1)/k]*k, dtype=float)
pi_xn = np.zeros((data_train.shape[0], k), dtype=float)
pn_i = np.zeros((k,data_train.shape[0]), dtype=float)

log_hood_criteria = 1e-3
max_iteration = 1000

L_last = 0
L = 1
iter = 0
#maybe accelerate through matrixation
while(L - L_last < log_hood_criteria or iter <= max_iteration):
    L_last = L
    sum = 0
    for n in range(data_train.shape[0]):
        sub_sum = 0
        for i in range(k):
#            a = np.exp(-0.5*((data_train[n]-m[i]).transpose().dot(np.linalg.inv(s[i]))).dot((data_train[n]-m[i])))
#            b = (np.linalg.det(s[i])**(-0.5))
            pi_xn[n,i] = pi[i]*np.exp(-0.5*((data_train[n]-m[i]).transpose().dot(np.linalg.inv(s[i]))).dot((data_train[n]-m[i])))*(np.linalg.det(s[i])**(-0.5))
#        a = pi_xn[n,:]
#        a = a/np.linalg.norm(a, ord=1)
        pi_xn[n,:] = pi_xn[n,:]/np.linalg.norm(pi_xn[n,:], ord=1)
        pn_i[:,n] = pi_xn[n,:]

    for i in range(k):
        pn_i[i,:] = pn_i[i,:]/np.linalg.norm(pn_i[i,:],ord=1)

    for i in range(k):
#        pn_i[i,:] = pn_i[i,:]/np.linalg.norm(pn_i[i,:],ord=1)
        temp_m = np.zeros((data_train[0].size),dtype=float)
        temp_s = np.zeros((data_train[0].size, data_train[0].size),dtype=float)
        for n in range(data_train.shape[0]):
            temp_m += pn_i[i,n] * data_train[n]
        m[i] = temp_m
        for n in range(data_train.shape[0]):
#            a = (data_train[n]-m[i]).dot((data_train[n]-m[i]).transpose())
            temp_s += pn_i[i,n] * np.dot(data_train[n]-m[i],(data_train[n]-m[i]).transpose())
        s[i] = temp_s
        pi[i] = float(np.sum(pi_xn[:,i]))/float(data_train.shape[0])

    L = 0
    for n in range(data_train.shape[0]):
        temp_L = 0
        for i in range(k):
            b = np.linalg.inv(s[i])
            a = np.exp(-0.5*((data_train[n]-m[i]).transpose().dot(np.linalg.inv(s[i]))).dot((data_train[n]-m[i])))
            temp_L += float(pi[i])/(np.sqrt(np.linalg.det(2*pi*s[i])))*np.exp(-0.5*((data_train[n]-m[i]).transpose().dot(np.linalg.inv(s[i]))).dot((data_train[n]-m[i])))
        L += np.log(temp_L)
    iter += 1
