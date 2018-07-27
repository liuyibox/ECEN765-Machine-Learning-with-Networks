import numpy as np

#############################[train data accuracy]#####################
label_data = {}
cur_label = 0
total_train_sample = 0
label_counts = []
array_list = []

path = 'trainingDigits/'+ str(0) + '_' + str(0) + '.txt'
lines = tuple(open(path, 'r'))
two_d_data = []
length = len(lines)
for i in range(length):
    two_d_data = two_d_data+[int(i) for i in (lines[i].split())[0]]
array_list.append(two_d_data)
array = np.array(two_d_data)


path = 'trainingDigits/'+ str(0) + '_' + str(1) + '.txt'
lines = tuple(open(path, 'r'))
two_d_data = []
length = len(lines)
for i in range(length):
    two_d_data = two_d_data+[int(i) for i in (lines[i].split())[0]]
array_list.append(two_d_data)
two_d_data = np.array(two_d_data)
array = np.concatenate((array[:,None], two_d_data[:,None]), axis=1)

while(cur_label != 10):
    file_exist = True

    label_count = 0

    while(file_exist):
        path = 'trainingDigits/'+ str(cur_label) + '_' + str(label_count) + '.txt'
        try:
            lines = tuple(open(path, 'r'))
            two_d_data = []
            length = len(lines)
            for i in range(length):
                two_d_data = two_d_data+[int(i) for i in (lines[i].split())[0]]
            array_list.append(two_d_data)
            two_d_data = np.array(two_d_data)
            array = np.concatenate((array, two_d_data[:,None]), axis=1)
            label_count = label_count + 1
            file_exist = True
        except IOError:
#            print label_count
#            label_data.update({cur_label : array})
            label_counts.append(label_count)
            file_exist = False
            cur_label = cur_label + 1
            break
    total_train_sample = total_train_sample + label_count


print total_train_sample
array = array[:,2:total_train_sample+3]
print array.shape

array_length = array.shape[0]
mean_vector  = np.array([np.mean(array[i,:])for i in range(array_length)])
print mean_vector.shape

array_list = zip(*array_list)
one_d_array_list = []
for i in range(array_length):
    one_d_array_list = one_d_array_list + [list(array_list[i])]

cov_mat = np.cov(one_d_array_list)

eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

#check if we get the right eigenvalue-eigenvector calculation
for i in range(len(eig_val_cov)):
    eigv = eig_vec_cov[:,i].reshape(1,1024).T
    np.testing.assert_array_almost_equal(cov_mat.dot(eigv),eig_val_cov[i]*eigv,decimal = 3, err_msg= ' ', verbose=True)

#check if each covariance vector eigen_vector is unit vector
#for ev in eig_vec_cov:
#    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

#transform the list with each element as (eigenvalue, eigenvector) pair
eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
eig_pairs.sort(key=lambda x:x[0], reverse=True)

#we get the first k = 50, 100, and 150 largest eigenvalues out
eigen_array_sequence_50 = ()
eigen_array_sequence_100 = ()
eigen_array_sequence_150 = ()
for i in range(150):

    if(i < 100):
        eigen_array_sequence_100 = eigen_array_sequence_100 + (eig_pairs[i][1].reshape(1024,1),)
        if(i < 50):
            eigen_array_sequence_50 = eigen_array_sequence_50 + (eig_pairs[i][1].reshape(1024,1),)

    eigen_array_sequence_150 = eigen_array_sequence_150 + (eig_pairs[i][1].reshape(1024,1),)

matrix_w_50 = np.hstack(eigen_array_sequence_50)
matrix_w_100 = np.hstack(eigen_array_sequence_100)
matrix_w_150 = np.hstack(eigen_array_sequence_150)

transformed_50 = matrix_w_50.T.dot(array)
assert transformed_50.shape == (50,total_train_sample), "The matrix is not 50x1934 dimensional."
transformed_100 = matrix_w_100.T.dot(array)
assert transformed_100.shape == (100,total_train_sample), "The matrix is not 100x1934 dimensional."
transformed_150 = matrix_w_150.T.dot(array)
assert transformed_150.shape == (150,total_train_sample), "The matrix is not 150x1934 dimensional."

#preprocess the PCA transformed to get 0/1 values
min_50 = np.amin(transformed_50,axis=1)
min_100 = np.amin(transformed_100,axis=1)
min_150 = np.amin(transformed_150,axis=1)

max_50 = np.amax(transformed_50,axis=1)
max_100 = np.amax(transformed_100,axis=1)
max_150 = np.amax(transformed_150,axis=1)

criterion50 = np.divide(np.add(min_50, max_50), 2.0)
criterion100 = np.divide(np.add(min_100, max_100), 2.0)
criterion150 = np.divide(np.add(min_150, max_150), 2.0)

for k in range(total_train_sample):
    for i in range(150):
        if(i < 100):
            transformed_100[i,k] = 1 if (transformed_100[i,k] > criterion100[i]) else 0
            if(i < 50):
                transformed_50[i, k] = 1 if (transformed_50[i, k] > criterion50[i]) else 0

        transformed_150[i, k] = 1 if (transformed_150[i, k] > criterion150[i]) else  0

#these are 0/1 values array
transformed_50 = np.array(transformed_50, dtype=int)
transformed_100 = np.array(transformed_100, dtype=int)
transformed_150 = np.array(transformed_150, dtype=int)

#we compute the mean of data here
mean_array_50 = np.zeros((50,10))
mean_array_100 = np.zeros((100,10))
mean_array_150 = np.zeros((150,10))
counted_sample = 0
for i in range(len(label_counts)):
    if i == 0:
        end_index = label_counts[i] + counted_sample
        label_transformed_data_50 = transformed_50[:, counted_sample:end_index ]
        label_transformed_data_100 = transformed_100[:, counted_sample:end_index ]
        label_transformed_data_150 = transformed_150[:, counted_sample:end_index ]
    else:
#        start_index = label_counts[i-1]
        end_index = label_counts[i] + counted_sample
        label_transformed_data_50 = transformed_50[:, counted_sample:end_index ]
        label_transformed_data_100 = transformed_100[:, counted_sample:end_index ]
        label_transformed_data_150 = transformed_150[:, counted_sample:end_index ]

    counted_sample += label_counts[i]
    mean_array_50[:,i] = label_transformed_data_50.mean(1)          #50*10
    mean_array_100[:,i] = label_transformed_data_100.mean(1)        #100*10
    mean_array_150[:,i] = label_transformed_data_150.mean(1)        #150*10


###################################[test data accuracy]##################################
#first, we use pca to transform test data
#label_data = {}
cur_label = 0
total_test_sample = 0
label_counts = []
array_list = []

path = 'testDigits/'+ str(0) + '_' + str(0) + '.txt'
lines = tuple(open(path, 'r'))
two_d_data = []
length = len(lines)
for i in range(length):
    two_d_data = two_d_data+[int(i) for i in (lines[i].split())[0]]
array_list.append(two_d_data)
array = np.array(two_d_data)


path = 'testDigits/'+ str(0) + '_' + str(1) + '.txt'
lines = tuple(open(path, 'r'))
two_d_data = []
length = len(lines)
for i in range(length):
    two_d_data = two_d_data+[int(i) for i in (lines[i].split())[0]]
array_list.append(two_d_data)
two_d_data = np.array(two_d_data)
array = np.concatenate((array[:,None], two_d_data[:,None]), axis=1)

while(cur_label != 10):
    file_exist = True

    label_count = 0

    while(file_exist):
        path = 'testDigits/'+ str(cur_label) + '_' + str(label_count) + '.txt'
        try:
            lines = tuple(open(path, 'r'))
            two_d_data = []
            length = len(lines)
            for i in range(length):
                two_d_data = two_d_data+[int(i) for i in (lines[i].split())[0]]
            #the data list rather than data array was collected due to the fact
            #we need to adopt the built in function to get one-d list to compute pca
            array_list.append(two_d_data)

            two_d_data = np.array(two_d_data)
            array = np.concatenate((array, two_d_data[:,None]), axis=1)
            label_count = label_count + 1
            file_exist = True
        except IOError:
#            print label_count
#            label_data.update({cur_label : array})
            label_counts.append(label_count)
            file_exist = False
            cur_label = cur_label + 1
            break
    total_test_sample = total_test_sample + label_count

print total_test_sample                     #946
array = array[:,2:total_test_sample+3]      #1024*946
print array.shape                           #1024*946

array_length = array.shape[0]
mean_vector  = np.array([np.mean(array[i,:])for i in range(array_length)])
print mean_vector.shape                     #1024*1

array_list = zip(*array_list)
one_d_array_list = []
for i in range(array_length):
    one_d_array_list = one_d_array_list + [list(array_list[i])]

cov_mat = np.cov(one_d_array_list)

eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

#check if we get the right eigenvalue-eigenvector calculation
for i in range(len(eig_val_cov)):
    eigv = eig_vec_cov[:,i].reshape(1,1024).T
    np.testing.assert_array_almost_equal(cov_mat.dot(eigv),eig_val_cov[i]*eigv,decimal = 3, err_msg= ' ', verbose=True)

#check if each covariance vector eigen_vector is unit vector
#for ev in eig_vec_cov:
#    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

#transform the list with each element as (eigenvalue, eigenvector) pair
eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
eig_pairs.sort(key=lambda x:x[0], reverse=True)

#we get the first k = 50, 100, and 150 largest eigenvalues out
eigen_array_sequence_50 = ()
eigen_array_sequence_100 = ()
eigen_array_sequence_150 = ()
for i in range(150):

    if(i < 100):
        eigen_array_sequence_100 = eigen_array_sequence_100 + (eig_pairs[i][1].reshape(1024,1),)
        if(i < 50):
            eigen_array_sequence_50 = eigen_array_sequence_50 + (eig_pairs[i][1].reshape(1024,1),)

    eigen_array_sequence_150 = eigen_array_sequence_150 + (eig_pairs[i][1].reshape(1024,1),)

matrix_w_50 = np.hstack(eigen_array_sequence_50)
matrix_w_100 = np.hstack(eigen_array_sequence_100)
matrix_w_150 = np.hstack(eigen_array_sequence_150)

transformed_50 = matrix_w_50.T.dot(array)
assert transformed_50.shape == (50,total_test_sample), "The matrix is not 50x946 dimensional."
transformed_100 = matrix_w_100.T.dot(array)
assert transformed_100.shape == (100,total_test_sample), "The matrix is not 100x946 dimensional."
transformed_150 = matrix_w_150.T.dot(array)
assert transformed_150.shape == (150,total_test_sample), "The matrix is not 150x946 dimensional."

#preprocess the PCA transformed to get 0/1 values
min_50 = np.amin(transformed_50,axis=1)
min_100 = np.amin(transformed_100,axis=1)
min_150 = np.amin(transformed_150,axis=1)

max_50 = np.amax(transformed_50,axis=1)
max_100 = np.amax(transformed_100,axis=1)
max_150 = np.amax(transformed_150,axis=1)

criterion50 = np.divide(np.add(min_50, max_50), 2.0)
criterion100 = np.divide(np.add(min_100, max_100), 2.0)
criterion150 = np.divide(np.add(min_150, max_150), 2.0)

for k in range(total_test_sample):
    for i in range(150):
        if(i < 100):
            transformed_100[i,k] = 1 if (transformed_100[i,k] > criterion100[i]) else 0
            if(i < 50):
                transformed_50[i, k] = 1 if (transformed_50[i, k] > criterion50[i]) else 0

        transformed_150[i, k] = 1 if (transformed_150[i, k] > criterion150[i]) else  0

#these are 0/1 values array for test data
transformed_50 = np.array(transformed_50, dtype=int)        #50*946
transformed_100 = np.array(transformed_100, dtype=int)      #100*946
transformed_150 = np.array(transformed_150, dtype=int)      #150*946


#nbc test accuracy
test_label_accuracy_array_50 = []
test_label_accuracy_array_100 = []
test_label_accuracy_array_150 = []
#label_accuracy_counts_array_50 = []
#label_accuracy_counts_array_100 = []
#label_accuracy_counts_array_150 = []
test_total_accuracy_50 = 0
test_total_accuracy_100 = 0
test_total_accuracy_150 = 0
test_counted_sample = 0
for i in range(10):
#for each label out of 10 labels, we compute the accuracy separately
    label_accuracy_50 = 0
    label_accuracy_100 = 0
    label_accuracy_150 = 0
    if i == 0:
        end_index = label_counts[i] + test_counted_sample
        label_transformed_data_50 = transformed_50[:, test_counted_sample:end_index ]           #50*label_counts[i] 0/1 values
        label_transformed_data_100 = transformed_100[:, test_counted_sample:end_index]          #100*label_counts[i] 0/1 values
        label_transformed_data_150 = transformed_150[:, test_counted_sample:end_index ]         #150*label_counts[i] 0/1 values
    else:
        #        start_index = label_counts[i-1]
        end_index = label_counts[i] + test_counted_sample
        label_transformed_data_50 = transformed_50[:, test_counted_sample:end_index ]
        label_transformed_data_100 = transformed_100[:, test_counted_sample:end_index ]
        label_transformed_data_150 = transformed_150[:, test_counted_sample:end_index ]

    test_counted_sample += label_counts[i]

    for test_sample_index in range(label_counts[i]):
        test_sample_prob_array_50 = []
        test_sample_prob_array_100 = []
        test_sample_prob_array_150 = []
        for k in range(10):                     #k=label_index
            target_class_predict_prob_50 = 0.0
            target_class_predict_prob_100 = 0.0
            target_class_predict_prob_150 = 0.0
            for j in range(150):

                if(j < 100):
                    if label_transformed_data_100[j,test_sample_index] == 1:
                        target_class_predict_prob_100 += np.log(mean_array_100[j,k]+ 0.0000000000000001)
                    else:
                        target_class_predict_prob_100 += np.log(1 - mean_array_100[j, k] + 0.0000000000000001)

                    if(j < 50):
                        if label_transformed_data_50[j,test_sample_index] == 1:
                            target_class_predict_prob_50 += np.log(mean_array_50[j,k]+ 0.0000000000000001)
                        else:
                            target_class_predict_prob_50 += np.log(1 - mean_array_50[j, k] + 0.0000000000000001)

                if label_transformed_data_150[j,test_sample_index] == 1:
                    target_class_predict_prob_150 += np.log(mean_array_150[j,k]+ 0.0000000000000001)
                else:
                    target_class_predict_prob_150 += np.log(1 - mean_array_150[j, k] + 0.0000000000000001)

            target_class_prob = label_counts[i] / (float(total_test_sample))                   #training sample label probability
            target_class_predict_prob_50 += np.log(target_class_prob)
            target_class_predict_prob_100 += np.log(target_class_prob)
            target_class_predict_prob_150 += np.log(target_class_prob)

            test_sample_prob_array_50.append(target_class_predict_prob_50)
            test_sample_prob_array_100.append(target_class_predict_prob_100)
            test_sample_prob_array_150.append(target_class_predict_prob_150)

        max_prob_index_50 = test_sample_prob_array_50.index(max(test_sample_prob_array_50))
        max_prob_index_100 = test_sample_prob_array_100.index(max(test_sample_prob_array_100))
        max_prob_index_150 = test_sample_prob_array_150.index(max(test_sample_prob_array_150))

        if max_prob_index_50 == i:
            label_accuracy_50 += 1
            test_total_accuracy_50 += 1
        if max_prob_index_100 == i:
            label_accuracy_100 += 1
            test_total_accuracy_100 += 1
        if max_prob_index_150 == i:
            label_accuracy_150 += 1
            test_total_accuracy_150 += 1

    test_label_accuracy_array_50.append(label_accuracy_50)
    test_label_accuracy_array_100.append(label_accuracy_100)
    test_label_accuracy_array_150.append(label_accuracy_150)

print "PCA to 50 dimensions: accuracy = ", test_total_accuracy_50
print "PCA to 100 dimensions: accuracy = ", test_total_accuracy_100
print "PCA to 150 dimensions: accuracy = ", test_total_accuracy_150