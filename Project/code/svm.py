import numpy as np
from sklearn.svm import SVC

path_train = 'production_rate_data.txt'
path_label = "label.txt"

data_train = []
data_label = []

try:
    with open(path_train) as f:
        for line in f:
            data_train.append([float(number) for number in line.split()])
    with open(path_label) as f:
        for line in f:
            data_label.append(int(line[0]))
            # data_label.append([float(number) for number in line.split()])

except IOError:
    print("error when reading train data\n")

data_train = np.array(data_train)
#label_train = data_train[:,0]
data_train = data_train.astype(np.int64)
#bias_train = np.ones((len(data_train),1))
#data_train = np.c_[data_train[:,1:35],bias_train]

data_label = np.array(data_label)
#label_test = data_test[:,0]
data_label = data_label.astype(np.int64)
#bias_test = np.ones((len(data_test),1))
#data_test = np.c_[data_test[:,1:35],bias_test]

data_test = data_train[100:199,:]
data_train = data_train[0:99,:]
label_test = data_label[100:199]
label_train = data_label[0:99]

#print(data_train.shape, data_test.shape)

clf = SVC(C=0.2, kernel='linear')
clf.fit(data_train, label_train)
#SVC( kernel='linear')

train_predict = np.array(clf.predict(data_train))
test_predict = np.array(clf.predict(data_test))
#print "train labels size: ", label_train.size
#print "test labels size: ", label_test.size
# print "train prediction size: ", train_predict.size
# print "test prediction size: ", test_predict.size

print "train error rates: ", (label_train != train_predict).sum()/(float(train_predict.size))
print "test error rates: ", (label_test != test_predict).sum()/(float(test_predict.size))
# print "number of support vectors: ", np.array(clf.support_vectors_).shape

