import numpy as np
import matplotlib.pyplot as plt

path_train = 'bclass-train'
path_test  = 'bclass-test'
data_train = []
data_test = []
#i = 0
try:
    with open(path_train) as f:
        for line in f:
            data_train.append([float(number) for number in line.split()])
#            data_train[i][0] = int (data_train[i][0])
#            i += 1
#        i = 0
    with open(path_test) as f:
        for line in f:
            data_test.append([float(number) for number in line.split()])
#            data_test[i][0] = int (data_train[i][0])
#            i += 1
except IOError:
    print "error when reading train data\n"

data_train = np.array(data_train)
label_train = data_train[:,0]
label_train = label_train.astype(np.int64)
bias_train = np.ones((len(data_train),1))
data_train = np.c_[data_train[:,1:35],bias_train]

data_test = np.array(data_test)
label_test = data_test[:,0]
label_test = label_test.astype(np.int64)
bias_test = np.ones((len(data_test),1))
data_test = np.c_[data_test[:,1:35],bias_test]

print data_train.shape, data_test.shape

train_data_mean = np.mean(data_train, axis=0)
test_data_mean = np.mean(data_test, axis=0)
train_data_mean = np.broadcast_to(train_data_mean,(len(data_train), len(train_data_mean)))
test_data_mean = np.broadcast_to(test_data_mean,(len(data_test), len(test_data_mean)))
#train_data_mean = np.repeat(train_data_mean,len(data_train),axis=0)
#test_data_mean = np.repeat(test_data_mean, len(data_test),axis=0)

#transform data to l1 unit norm
L1_norm = np.zeros(len(data_train[0]), dtype=float)
data_train = data_train - train_data_mean
for i in range(len(data_train[0])):
    L1_norm[i] = np.linalg.norm(data_train[:,i], ord = 1)/float(len(data_train))
    if L1_norm[i] != 0:
        data_train[:,i] = data_train[:,i]/L1_norm[i]

L1_norm = np.zeros(len(data_test[0]), dtype=float)
data_test = data_test - test_data_mean
for i in range(len(data_test[0])):
    L1_norm[i] = np.linalg.norm(data_test[:,i], ord = 1)/float(len(data_test))
    if L1_norm[i] != 0:
        data_test[:,i] = data_test[:,i]/L1_norm[i]

theta_train = np.zeros(35)
converged = False
train_accuracy = 0
train_iter = 1000
error_train = []
error_test = []
while(converged != True and train_iter > 0):
    converged = True
    train_accuracy = len(data_train)
    for i in range(len(data_train)):
        inner_product = np.inner(data_train[i],theta_train)
        result = int(np.sign(inner_product))
        if result != label_train[i]:
            theta_train = theta_train + label_train[i]*data_train[i]
            converged = False
            train_accuracy -= 1
    error_train.append(1-train_accuracy/float(len(data_train)))

    test_accuracy = len(data_test)
    for i in range(len(data_test)):
        inner_product = np.inner(data_test[i], theta_train)
        result = int(np.sign(inner_product))
        if result != label_test[i]:
            test_accuracy -= 1
    error_test.append(1-test_accuracy/float(len(data_test)))

    train_iter -= 1
print train_accuracy/float(len(data_train))

test_accuracy = len(data_test)
for i in range(len(data_test)):
    inner_product = np.inner(data_test[i], theta_train)
    result = int(np.sign(inner_product))
    if result != label_test[i]:
        test_accuracy -= 1
print test_accuracy/float(len(data_test))

k = range(1,len(error_test)+1)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_xlabel("Perceptron_L1")
ax1.plot(k,error_train, 'r', label= "train error")
ax1.plot(k,error_test, 'b', label = "test error")
plt.show()