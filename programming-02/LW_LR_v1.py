import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def sigmoid(x): return 1/(1+np.exp(-x))

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
    print "error when reading train data\n"

data_train = np.array(data_train)
label_train = data_train[:,0]
label_train = label_train.astype(np.int64)
label_train_01 = label_train
for i in range(len(label_train)):
    if label_train[i] == -1:
        label_train_01[i] = 0
bias_train = np.ones((len(data_train),1))
data_train = np.c_[data_train[:,1:35],bias_train]

data_test = np.array(data_test)
label_test = data_test[:,0]
label_test = label_test.astype(np.int64)
label_test_01 = label_train
for i in range(len(label_test)):
    if label_test[i] == -1:
        label_test_01[i] = 0
bias_test = np.ones((len(data_test),1))
data_test = np.c_[data_test[:,1:35],bias_test]

print data_train.shape, data_test.shape

x_outer = np.zeros((len(data_train),35, 35 ))
for i in range(len(data_train)):
    x_outer[i] = np.outer(data_train[i], data_train[i])

eta = 0.01
lamta = 0.001
weight_i = np.zeros(35, dtype=np.float64)
H_sum1 = np.zeros((35, 35), dtype=np.float64)
G_sum1 = np.zeros(35, dtype=np.float64)
identity_matrix = np.identity(35,dtype=np.float64)

tao_array = [0.01,0.05,0.1,0.5,1.0,5.0]
beta_train0 = np.random.normal(0.0, 1, 35)
error = []
for tao in tao_array:
    test_accuracy = len(data_test)
    converged = False
    beta_train = beta_train0
    for i in range(len(data_test)):
        train_iter = 1000
        cur_data_test = data_test[i]            #one array of length at 34
        weight_i = np.exp(-(LA.norm(data_train - cur_data_test,axis=1)/(2*tao*tao)))
        while(converged != True and train_iter > 0):
            converged = True
            inner_product = data_train.dot(beta_train)
            temp_sigmoid = sigmoid(inner_product)

            result = (np.sign(inner_product)).astype(int)
            if not np.array_equal(result, label_train):
                converged = False
#            for j in range(len(data_train)):
#                result = int(np.sign(inner_product[j]))
#                if result != label_train[j]:
#                    converged = False
            if converged == False:
                H_sum1 = np.tensordot(np.multiply(np.multiply(weight_i, temp_sigmoid), (1-temp_sigmoid)),x_outer, axes=(0,0))
                G_sum1 = np.tensordot(np.multiply(weight_i,temp_sigmoid-label_train_01),data_train, axes=(0,0))
                H_sum1 -= 2*lamta * identity_matrix
                G_sum1 -= 2*lamta * beta_train
                beta_train -= eta*H_sum1.dot(G_sum1)
            train_iter -= 1

        inner_product = np.inner(cur_data_test, beta_train)
        result = int(np.sign(inner_product))
        if result != label_test[i]:
            test_accuracy -= 1
    error.append(1-test_accuracy/float(len(data_test)))
    print tao, test_accuracy/float(len(data_test))


fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_xlabel("Newton's method")
ax1.plot(tao_array,error, 'r', label= "error")
plt.show()
