import numpy as np
#import math
import collections
import bisect


def get_sorted_pair(dist_array, label_counts, cur_target_label_index):
    counted_sample = 0
    k_distance_size = 0
    cur_max_distance = 0
    k_sorted_distance = []
    k_distance_pair = []

    for label in range(10):
        end_index = label_counts[label] + counted_sample
        for cur_label_sample_index in range(counted_sample, end_index):
            if cur_label_sample_index == cur_target_label_index:
                continue

            temp_distance = dist_array[cur_label_sample_index]
            if ((temp_distance < cur_max_distance) and k_distance_size == 10):
                for dist, l in k_distance_pair:
                    if dist == cur_max_distance:
                        k_distance_pair.remove((dist, l))
                        break

                k_distance_pair.append((temp_distance, label))
                del k_sorted_distance[-1]
                bisect.insort(k_sorted_distance, temp_distance)
                cur_max_distance = k_sorted_distance[-1]

            if k_distance_size < 10:
                k_distance_pair.append((temp_distance, label))
                bisect.insort(k_sorted_distance, temp_distance)
                cur_max_distance = k_sorted_distance[-1]
                k_distance_size += 1

        counted_sample += label_counts[label]

    return sorted(k_distance_pair, key=lambda x: x[0])



##############################[ KNN with PCA ]#############################
#############################[ train data accuracy ]#####################
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


#train error
knn_50 = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
knn_100 = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
knn_150 = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}

# #for k in [i+1 for i in range(10)]:
# for k in [10]:
cur_label = 0
total_accuracy_50 = 0
total_accuracy_100 = 0
total_accuracy_150 = 0
label_accuracy_array_50 = []
label_accuracy_array_100 = []
label_accuracy_array_150 = []
train_label_counts_array = []
#the plan is for each label, we directly compute the norm of each element
#with current sample

distance_matrix_50 = np.zeros((total_train_sample, total_train_sample),dtype=float)
distance_matrix_100 = np.zeros((total_train_sample, total_train_sample),dtype=float)
distance_matrix_150 = np.zeros((total_train_sample, total_train_sample),dtype=float)
for sample_index in range(total_train_sample):    #extract the total train size 1934
#   next_sample_index = sample_index + 1
    for next_sample_index in range(sample_index+1, total_train_sample):

        distance_matrix_50[sample_index, next_sample_index] = \
            np.linalg.norm(transformed_50[:,sample_index] - transformed_50[:,next_sample_index])
        distance_matrix_50[next_sample_index, sample_index] = distance_matrix_50[sample_index, next_sample_index]

        distance_matrix_100[sample_index, next_sample_index] = \
            np.linalg.norm(transformed_100[:, sample_index] - transformed_100[:, next_sample_index])
        distance_matrix_100[next_sample_index, sample_index] = distance_matrix_100[sample_index, next_sample_index]

        distance_matrix_150[sample_index, next_sample_index] = \
            np.linalg.norm(transformed_150[:, sample_index] - transformed_150[:, next_sample_index])
        distance_matrix_150[next_sample_index, sample_index] = distance_matrix_150[sample_index, next_sample_index]

target_counted_sample = 0
counted_sample = 0
k_sorted_distance_50 = []
k_sorted_distance_100 = []
k_sorted_distance_150 = []
k_distance_size_50 = 0
k_distance_size_100 = 0
k_distance_size_150 = 0
cur_max_distance_50 = 0.0
cur_max_distance_100 = 0.0
cur_max_distance_150 = 0.0

for target_label in range(10):
    target_end_index = label_counts[target_label] + target_counted_sample
    for cur_target_label_index in range(target_counted_sample, target_end_index):

        cur_distance_array_50 = distance_matrix_50[cur_target_label_index, :]
        cur_distance_array_100 = distance_matrix_100[cur_target_label_index, :]
        cur_distance_array_150 = distance_matrix_150[cur_target_label_index, :]

        k_distance_pair_50 = get_sorted_pair(cur_distance_array_50, label_counts, cur_target_label_index)
        k_distance_pair_100 = get_sorted_pair(cur_distance_array_100, label_counts, cur_target_label_index)
        k_distance_pair_150 = get_sorted_pair(cur_distance_array_150, label_counts, cur_target_label_index)

        for i in range(10):
            result_50 = [0]*10
            result_100 = [0]*10
            result_150 = [0]*10

            for j in range(i+1):
                temp_item_50 = k_distance_pair_50[j]
                temp_item_100 = k_distance_pair_100[j]
                temp_item_150 = k_distance_pair_150[j]

                result_50[temp_item_50[1]] += 1
                result_100[temp_item_100[1]] += 1
                result_150[temp_item_150[1]] += 1

            prediction_label_50 = result_50.index(max(result_50))
            prediction_label_100 = result_100.index(max(result_100))
            prediction_label_150 = result_150.index(max(result_150))

            if prediction_label_50 == target_label:
                knn_50[i+1] = knn_50[i+1] + 1
            if prediction_label_100 == target_label:
                knn_100[i+1] = knn_100[i+1] + 1
            if prediction_label_150 == target_label:
                knn_150[i+1] = knn_150[i+1] + 1

    target_counted_sample += label_counts[target_label]
print "knn train accuracy (50 dimension) --> ", knn_50
print "knn train accuracy (100 dimension) --> ", knn_100
print "knn train accuracy (150 dimension) --> ", knn_150


##############################[ KNN with PCA ]#############################
#############################[ test data accuracy ]########################
label_data = {}
cur_label = 0
total_test_sample = 0
test_label_counts = []
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
            array_list.append(two_d_data)
            two_d_data = np.array(two_d_data)
            array = np.concatenate((array, two_d_data[:,None]), axis=1)
            label_count = label_count + 1
            file_exist = True
        except IOError:
#            print label_count
#            label_data.update({cur_label : array})
            test_label_counts.append(label_count)
            file_exist = False
            cur_label = cur_label + 1
            break
    total_test_sample = total_test_sample + label_count


print total_test_sample
array = array[:,2:total_test_sample+3]
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

test_transformed_50 = matrix_w_50.T.dot(array)
assert test_transformed_50.shape == (50,total_test_sample), "The matrix is not 50x1934 dimensional."
test_transformed_100 = matrix_w_100.T.dot(array)
assert test_transformed_100.shape == (100,total_test_sample), "The matrix is not 100x1934 dimensional."
test_transformed_150 = matrix_w_150.T.dot(array)
assert test_transformed_150.shape == (150,total_test_sample), "The matrix is not 150x1934 dimensional."

#preprocess the PCA transformed to get 0/1 values
min_50 = np.amin(test_transformed_50,axis=1)
min_100 = np.amin(test_transformed_100,axis=1)
min_150 = np.amin(test_transformed_150,axis=1)

max_50 = np.amax(test_transformed_50,axis=1)
max_100 = np.amax(test_transformed_100,axis=1)
max_150 = np.amax(test_transformed_150,axis=1)

criterion50 = np.divide(np.add(min_50, max_50), 2.0)
criterion100 = np.divide(np.add(min_100, max_100), 2.0)
criterion150 = np.divide(np.add(min_150, max_150), 2.0)

for k in range(total_test_sample):
    for i in range(150):
        if(i < 100):
            test_transformed_100[i,k] = 1 if (test_transformed_100[i,k] > criterion100[i]) else 0
            if(i < 50):
                test_transformed_50[i, k] = 1 if (test_transformed_50[i, k] > criterion50[i]) else 0

        test_transformed_150[i, k] = 1 if (test_transformed_150[i, k] > criterion150[i]) else  0

#these are 0/1 values array
test_transformed_50 = np.array(test_transformed_50, dtype=int)      #50*946
test_transformed_100 = np.array(test_transformed_100, dtype=int)    #100*946
test_transformed_150 = np.array(test_transformed_150, dtype=int)    #150*946

#train error
knn_50 = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
knn_100 = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
knn_150 = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
#
# # #for k in [i+1 for i in range(10)]:
# # for k in [10]:
test_cur_label = 0
test_total_accuracy_50 = 0
test_total_accuracy_100 = 0
test_total_accuracy_150 = 0
test_label_accuracy_array_50 = []
test_label_accuracy_array_100 = []
test_label_accuracy_array_150 = []
test_train_label_counts_array = []
# #the plan is for each label, we directly compute the norm of each element
# #with current sample
#
test_distance_matrix_50 = np.zeros((total_test_sample, total_train_sample),dtype=float)
test_distance_matrix_100 = np.zeros((total_test_sample, total_train_sample),dtype=float)
test_distance_matrix_150 = np.zeros((total_test_sample, total_train_sample),dtype=float)
for test_sample_index in range(total_test_sample):    #extract the total train size 1934
#   next_sample_index = sample_index + 1
    for next_sample_index in range(total_train_sample):

        test_distance_matrix_50[test_sample_index, next_sample_index] = \
            np.linalg.norm(transformed_50[:,test_sample_index] - transformed_50[:,next_sample_index])
#        test_distance_matrix_50[next_sample_index, sample_index] = test_distance_matrix_50[sample_index, next_sample_index]

        test_distance_matrix_100[test_sample_index, next_sample_index] = \
            np.linalg.norm(transformed_100[:, test_sample_index] - transformed_100[:, next_sample_index])
#        test_distance_matrix_100[next_sample_index, sample_index] = test_distance_matrix_100[sample_index, next_sample_index]

        test_distance_matrix_150[test_sample_index, next_sample_index] = \
            np.linalg.norm(transformed_150[:, test_sample_index] - transformed_150[:, next_sample_index])
#        test_distance_matrix_150[next_sample_index, sample_index] = test_distance_matrix_150[sample_index, next_sample_index]

#  now the 3 matrices should be 946*1934 dimensional

target_counted_sample = 0
counted_sample = 0
k_sorted_distance_50 = []
k_sorted_distance_100 = []
k_sorted_distance_150 = []
k_distance_size_50 = 0
k_distance_size_100 = 0
k_distance_size_150 = 0
cur_max_distance_50 = 0.0
cur_max_distance_100 = 0.0
cur_max_distance_150 = 0.0
#
for target_label in range(10):
    target_end_index = test_label_counts[target_label] + target_counted_sample
    for cur_target_label_index in range(target_counted_sample, target_end_index):

        cur_distance_array_50 = test_distance_matrix_50[cur_target_label_index, :]
        cur_distance_array_100 = test_distance_matrix_100[cur_target_label_index, :]
        cur_distance_array_150 = test_distance_matrix_150[cur_target_label_index, :]

        k_distance_pair_50 = get_sorted_pair(cur_distance_array_50, test_label_counts, cur_target_label_index)
        k_distance_pair_100 = get_sorted_pair(cur_distance_array_100, test_label_counts, cur_target_label_index)
        k_distance_pair_150 = get_sorted_pair(cur_distance_array_150, test_label_counts, cur_target_label_index)
#
        for i in range(10):
            result_50 = [0]*10
            result_100 = [0]*10
            result_150 = [0]*10

            for j in range(i+1):
                temp_item_50 = k_distance_pair_50[j]
                temp_item_100 = k_distance_pair_100[j]
                temp_item_150 = k_distance_pair_150[j]

                result_50[temp_item_50[1]] += 1
                result_100[temp_item_100[1]] += 1
                result_150[temp_item_150[1]] += 1

            prediction_label_50 = result_50.index(max(result_50))
            prediction_label_100 = result_100.index(max(result_100))
            prediction_label_150 = result_150.index(max(result_150))

            if prediction_label_50 == target_label:
                knn_50[i+1] = knn_50[i+1] + 1
            if prediction_label_100 == target_label:
                knn_100[i+1] = knn_100[i+1] + 1
            if prediction_label_150 == target_label:
                knn_150[i+1] = knn_150[i+1] + 1
#
    target_counted_sample += test_label_counts[target_label]

print "knn testing accuracy (50 dimension) --> ", knn_50
print "knn testing accuracy (100 dimension) --> ", knn_100
print "knn testing accuracy (150 dimension) --> ", knn_150