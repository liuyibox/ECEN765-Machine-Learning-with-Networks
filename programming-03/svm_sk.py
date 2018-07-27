import numpy as np
from sklearn.svm import SVC

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
    print("error when reading train data\n")

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

#print(data_train.shape, data_test.shape)

clf = SVC(C=4, kernel='rbf',gamma=4)
clf.fit(data_train, label_train)
#SVC( kernel='linear')

train_predict = np.array(clf.predict(data_train))
test_predict = np.array(clf.predict(data_test))
print "train labels size: ", label_train.size
print "test labels size: ", label_test.size
print "train prediction size: ", train_predict.size
print "test prediction size: ", test_predict.size

print "train error rates: ", (label_train != train_predict).sum()/(float(train_predict.size))
print "test error rates: ", (label_test != test_predict).sum()/(float(test_predict.size))
print "number of support vectors: ", np.array(clf.support_vectors_).shape

