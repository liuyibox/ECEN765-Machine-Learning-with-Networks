import numpy as np

#pre_label = 0
cur_label = 0
total_train_sample = 0
two_d_array = []
label_counts = []
while(cur_label != 10):
    file_exist = True
    trained_2d = []
    label_count = 0
    array = []
    while(file_exist):
        path = 'trainingDigits/'+ str(cur_label) + '_' + str(label_count) + '.txt'
        try:
            lines = tuple(open(path, 'r'))
            two_d_data = []
            length = len(lines)
            for i in range(length):
                two_d_data.append([int(i) for i in (lines[i].split())[0]])
            array.append(two_d_data)
            label_count = label_count + 1
            file_exist = True
        except IOError:
            print label_count
            label_counts.append(label_count)
            matrix = np.array(array)
            summed_2d_matrix = np.sum(matrix, axis=0)
            trained_2d = [[float(j) / label_count for j in i] for i in summed_2d_matrix]
            file_exist = False
            cur_label = cur_label + 1
            break
    two_d_array.append(trained_2d)
    total_train_sample = total_train_sample + label_count

label_counts = [float(i)/total_train_sample for i in label_counts]
print total_train_sample



#train accuracy
cur_label = 0
total_accuracy = 0
label_accuracy_array = []
train_label_counts_array = []
while(cur_label != 10):
    file_exist = True
    label_count = 0
    label_accuracy = 0
    while(file_exist):
        path = 'trainingDigits/' + str(cur_label) + '_' + str(label_count) + '.txt'
        try:
            lines = tuple(open(path, 'r'))
            two_d_data = []
            length = len(lines)
            for i in range(length):
                two_d_data.append([int(i) for i in (lines[i].split())[0]])

            train_sample_prob_array = []

            for k in range(10):
                target_class_array = two_d_array[k]
                target_class_prob = label_counts[k]                     #training sample label probability
            #   train_sample_pr = np.zeros(shape=(32, 32), dtype=float)
                target_class_predict_prob = 0.0
                for i in range(32):
                    for j in range(32):
                        if two_d_data[i][j] == 1:
                            target_class_predict_prob += np.log(target_class_array[i][j] + 0.0000000000000001)
                        else:
                            target_class_predict_prob += np.log(1 - target_class_array[i][j] + 0.0000000000000001)
                target_class_predict_prob += np.log(target_class_prob)
                train_sample_prob_array.append(target_class_predict_prob)

            max_prob_index = train_sample_prob_array.index(max(train_sample_prob_array))
            if max_prob_index == cur_label:
                label_accuracy += 1
                total_accuracy += 1
            label_count += 1

        except IOError:
            print label_count
            label_accuracy_array.append(label_accuracy)
            train_label_counts_array.append(label_count)
            file_exist = False
            cur_label = cur_label + 1
            break

print label_accuracy_array
totoal_train_sample = np.sum([train_label_counts_array])
print "train accuracy = " + str(float(total_accuracy)/totoal_train_sample)
print total_accuracy


#test accuracy
cur_label = 0
total_accuracy = 0
label_accuracy_array = []
test_label_counts_array = []
while(cur_label != 10):
    file_exist = True
    label_count = 0
    label_accuracy = 0
    while(file_exist):
        path = 'testDigits/' + str(cur_label) + '_' + str(label_count) + '.txt'
        try:
            lines = tuple(open(path, 'r'))
            two_d_data = []
            length = len(lines)
            for i in range(length):
                two_d_data.append([int(i) for i in (lines[i].split())[0]])

            test_sample_prob_array = []

            for k in range(10):
                target_class_array = two_d_array[k]
                target_class_prob = label_counts[k]                     #training sample label probability
            #    test_sample_pr = np.zeros(shape=(32, 32), dtype=float)
                target_class_predict_prob = 0.0
                for i in range(32):
                    for j in range(32):
                        if two_d_data[i][j] == 1:
                            target_class_predict_prob += np.log(target_class_array[i][j] + 0.0000000000000001)
                        else:
                            target_class_predict_prob += np.log(1 - target_class_array[i][j] + 0.0000000000000001)
                target_class_predict_prob += np.log(target_class_prob)
                test_sample_prob_array.append(target_class_predict_prob)

            max_prob_index = test_sample_prob_array.index(max(test_sample_prob_array))
            if max_prob_index == cur_label:
                label_accuracy += 1
                total_accuracy += 1
            label_count += 1

        except IOError:
            print label_count
            label_accuracy_array.append(label_accuracy)
            test_label_counts_array.append(label_count)
            file_exist = False
            cur_label = cur_label + 1
            break

print label_accuracy_array
totoal_test_sample = np.sum([test_label_counts_array])
print "test accuracy = " + str(float(total_accuracy)/totoal_test_sample)
print total_accuracy
