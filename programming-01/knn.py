import numpy as np
#import math
import collections
import bisect

label_data = {}
cur_label = 0
total_train_sample = 0
#two_d_array = []
label_counts = []
while(cur_label != 10):
    file_exist = True
#    trained_2d = []
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
            label_data.update({cur_label : array})
            label_counts.append(label_count)
#            matrix = np.array(array)
#            summed_2d_matrix = np.sum(matrix, axis=0)
#            trained_2d = [[float(j) / label_count for j in i] for i in summed_2d_matrix]
            file_exist = False
            cur_label = cur_label + 1
            break
#    two_d_array.append(trained_2d)
    total_train_sample = total_train_sample + label_count

#label_counts = [float(i)/total_train_sample for i in label_counts]
print total_train_sample


#train error
knn = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
# #for k in [i+1 for i in range(10)]:
for k in [10]:
    cur_label = 0
    total_accuracy = 0
    label_accuracy_array = []
    test_label_counts_array = []
    while(cur_label != 10):
        file_exist = True
        label_count = 0
        label_accuracy = 0
        while(file_exist):
            path = 'trainingDigits/' + str(cur_label) + '_' + str(label_count) + '.txt'
            try:
                lines = tuple(open(path,'r'))
                two_d_data = []
                length = len(lines)
                for i in range(length):
                    two_d_data.append([int(i) for i in (lines[i].split())[0]])

#                k_distance_pair = {}
#                k_distance_pair = collections.defaultdict(list)
                k_sorted_distance = []
                k_distance_pair = []
                k_distance_size = 0
                cur_max_distance = 0.0
                for label_index in range(10):
                    cur_train_data = label_data.get(label_index)
                    cur_train_data_size = len(cur_train_data)
                    for cur_train_data_index in range(cur_train_data_size):
                        two_d_train_data = cur_train_data[cur_train_data_index]
                        temp_distance = np.linalg.norm(np.array(two_d_train_data) - np.array(two_d_data))
                        # for i in range(32):
                        #     for j in range(32):
                        #         if two_d_train_data[i][j] == two_d_data[i][j]:
                        #             temp_distance += 1

                        if (temp_distance < cur_max_distance) and k_distance_size == k:
                        #    del k_distance_pair[cur_max_distance]

                            for dist, l in k_distance_pair:
                                if dist == cur_max_distance:
                                    k_distance_pair.remove((dist, l))
                                    break

                        #    k_distance_pair = [(dist,l) for dist, l in k_distance_pair if dist != cur_max_distance]
                            k_distance_pair.append((temp_distance, label_index))
                            del k_sorted_distance[-1]
                            bisect.insort(k_sorted_distance, temp_distance)
                            cur_max_distance = k_sorted_distance[-1]


                        if k_distance_size < k:
                            k_distance_pair.append((temp_distance, label_index))
                            bisect.insort(k_sorted_distance,temp_distance)
                        #    if temp_distance > cur_max_distance:
                        #        cur_max_distance = temp_distance
                            cur_max_distance = k_sorted_distance[-1]
                            k_distance_size += 1

                k_distance_pair = sorted(k_distance_pair, key=lambda x: x[0])
#                od = collections.OrderedDict(sorted(k_distance_pair.items()))
                for i in range(10):

                    result = [0] * 10
                    for j in range(i+1):
                        temp_item = k_distance_pair[j]
                        result[temp_item[1]] += 1

                    prediction_label = result.index(max(result))
                    if prediction_label == cur_label:
                        knn[i+1] = knn[i+1]+1
                #        label_accuracy += 1
                #        total_accuracy += 1

                label_count += 1

            #    for dist_key, dist_value in k_distance_pair.iteritems():
     #           print "current label = " + str(cur_label)+ ", prediction label = " + str(prediction_label)

            except IOError:
                print label_count
                label_accuracy_array.append(label_accuracy)
                test_label_counts_array.append(label_count)
                file_exist = False
                cur_label += 1
                break
print "knn training accuracy --> ", knn

#
#test error
knn = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
#for k in [i+1 for i in range(10)]:
for k in [10]:
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
                lines = tuple(open(path,'r'))
                two_d_data = []
                length = len(lines)
                for i in range(length):
                    two_d_data.append([int(i) for i in (lines[i].split())[0]])

#                k_distance_pair = {}
#                k_distance_pair = collections.defaultdict(list)
                k_sorted_distance = []
                k_distance_pair = []
                k_distance_size = 0
                cur_max_distance = 0.0
                for label_index in range(10):
                    cur_train_data = label_data.get(label_index)
                    cur_train_data_size = len(cur_train_data)
                    for cur_train_data_index in range(cur_train_data_size):
                        two_d_train_data = cur_train_data[cur_train_data_index]
                        temp_distance = np.linalg.norm(np.array(two_d_train_data) - np.array(two_d_data))
                        # for i in range(32):
                        #     for j in range(32):
                        #         if two_d_train_data[i][j] == two_d_data[i][j]:
                        #             temp_distance += 1

                        if (temp_distance < cur_max_distance) and k_distance_size == k:
                        #    del k_distance_pair[cur_max_distance]

                            for dist, l in k_distance_pair:
                                if dist == cur_max_distance:
                                    k_distance_pair.remove((dist, l))
                                    break

                        #    k_distance_pair = [(dist,l) for dist, l in k_distance_pair if dist != cur_max_distance]
                            k_distance_pair.append((temp_distance, label_index))
                            del k_sorted_distance[-1]
                            bisect.insort(k_sorted_distance, temp_distance)
                            cur_max_distance = k_sorted_distance[-1]


                        if k_distance_size < k:
                            k_distance_pair.append((temp_distance, label_index))
                            bisect.insort(k_sorted_distance,temp_distance)
                        #    if temp_distance > cur_max_distance:
                        #        cur_max_distance = temp_distance
                            cur_max_distance = k_sorted_distance[-1]
                            k_distance_size += 1

                k_distance_pair = sorted(k_distance_pair, key=lambda x: x[0])
#                od = collections.OrderedDict(sorted(k_distance_pair.items()))
                for i in range(10):

                    result = [0] * 10
                    for j in range(i+1):
                        temp_item = k_distance_pair[j]
                        result[temp_item[1]] += 1

                    prediction_label = result.index(max(result))
                    if prediction_label == cur_label:
                        knn[i+1] = knn[i+1]+1
                #        label_accuracy += 1
                #        total_accuracy += 1

                label_count += 1

            #    for dist_key, dist_value in k_distance_pair.iteritems():
     #           print "current label = " + str(cur_label)+ ", prediction label = " + str(prediction_label)

            except IOError:
                print label_count
                label_accuracy_array.append(label_accuracy)
                test_label_counts_array.append(label_count)
                file_exist = False
                cur_label += 1
                break
print "knn testing accuracy --> ", knn

# #test error
# knn = {}
# for k in [i+1 for i in range(10)]:
#     cur_label = 0
#     total_accuracy = 0
#     label_accuracy_array = []
#     test_label_counts_array = []
#     while(cur_label != 10):
#         file_exist = True
#         label_count = 0
#         label_accuracy = 0
#         while(file_exist):
#             path = 'testDigits/' + str(cur_label) + '_' + str(label_count) + '.txt'
#             try:
#                 lines = tuple(open(path,'r'))
#                 two_d_data = []
#                 length = len(lines)
#                 for i in range(length):
#                     two_d_data.append([int(i) for i in (lines[i].split())[0]])
#
#                 k_distance_pair = {}
#                 k_distance_size = 0
#                 cur_min_distance = float('inf')
#                 for label_index in range(10):
#                     cur_train_data = label_data.get(label_index)
#                     cur_train_data_size = len(cur_train_data)
#                     for cur_train_data_index in range(cur_train_data_size):
#                         two_d_train_data = cur_train_data[cur_train_data_index]
#                         temp_distance = np.linalg.norm(np.array(two_d_train_data) - np.array(two_d_data))
#                         # for i in range(32):
#                         #     for j in range(32):
#                         #         if two_d_train_data[i][j] == two_d_data[i][j]:
#                         #             temp_distance += 1
#
#                         if temp_distance < cur_min_distance:
#                             if k_distance_size == k:
#                                 del k_distance_pair[cur_min_distance]
#                                 k_distance_pair.update({temp_distance:label_index})
#                                 cur_min_distance = temp_distance
#                             else:
#                                 k_distance_pair.update({temp_distance:label_index})
#                                 cur_min_distance = temp_distance
#                                 k_distance_size += 1
#
#                 result = [0] * 10
#                 for dist_key, dist_value in k_distance_pair.iteritems():
#                     result[dist_value] += 1
#
#                 prediction_label = result.index(max(result))
#                 if prediction_label == cur_label:
#                     label_accuracy += 1
#                     total_accuracy += 1
#                 label_count += 1
#
#      #           print "current label = " + str(cur_label)+ ", prediction label = " + str(prediction_label)
#
#             except IOError:
#                 print label_count
#                 label_accuracy_array.append(label_accuracy)
#                 test_label_counts_array.append(label_count)
#                 file_exist = False
#                 cur_label += 1
#                 break
#
#     total_test_sample = np.sum([test_label_counts_array])
#     accuracy_percent = (float(total_accuracy)/total_test_sample)
#     print "test accuracy = " + str(float(total_accuracy)/total_test_sample)
#     print label_accuracy_array
#     print total_accuracy
#     knn.update({k:accuracy_percent})