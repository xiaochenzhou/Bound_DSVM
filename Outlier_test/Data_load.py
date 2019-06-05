import numpy as np
from sklearn import svm, datasets
from Data_dist import data_partition, kmeans_partition
import Edge_node as Ed
import sklearn
import random_select as Rs
import Central_node as Ce
from sklearn.utils import shuffle
from sklearn import preprocessing
import copy
import matplotlib.pyplot as plt
from plot_utility import SVM_plot
import csv
from sklearn import preprocessing


def read_UCI_data(num):
    if (num == 0):
        Training_set = sklearn.datasets.load_svmlight_file('skin_nonskin.txt')
    elif (num == 1):
        Training_set = sklearn.datasets.load_svmlight_file('phishing.txt')
    elif (num == 2):
        Training_set = sklearn.datasets.load_svmlight_file('mushrooms.txt')
    elif (num == 3):
        Training_set = sklearn.datasets.load_svmlight_file('ijcnn1.tr')
    elif (num == 4):
        Training_set = sklearn.datasets.load_svmlight_file('mnist.scale')
    else:
        raise ValueError


    Train_label_t = Training_set[1]
    Train_label_t = (2 * (Train_label_t % 2)) - 1
    Train_data_t = np.array(Training_set[0].todense())
    Train_data_t = preprocessing.normalize(Train_data_t)

    Train_data_s, Train_label_s = shuffle(Train_data_t, Train_label_t, random_state = 1)

    Train_data_s, Train_label_s = data_clean(Train_data_s, Train_label_s)

    Train_label = Train_label_s[:10000]
    Train_data = Train_data_s[:10000, :]

    g_size = np.size(Train_label)

    Test_data = Train_data[int(g_size * 4 / 5):g_size]
    Test_label = Train_label[int(g_size * 4 / 5):g_size]
    Train_data = Train_data[:int(g_size * 4 / 5)]
    Train_label = Train_label[:int(g_size * 4 / 5)]

    return Train_data, Train_label, Test_data, Test_label

def read_skin():
    Train_data_t = np.load("skin_nonskin_data.npy")
    Train_label_t = np.load("skin_nonskin_label.npy")

    Train_data_t = preprocessing.normalize(Train_data_t)


    Train_label_t = (2 * (Train_label_t % 2)) - 1

    Train_data_s, Train_label_s = shuffle(Train_data_t, Train_label_t, random_state=1)

    Train_label = Train_label_s[:10000]
    Train_data = Train_data_s[:10000, :]

    g_size = np.size(Train_label)

    Test_data = Train_data[int(g_size * 4 / 5):g_size]
    Test_label = Train_label[int(g_size * 4 / 5):g_size]
    Train_data = Train_data[:int(g_size * 4 / 5)]
    Train_label = Train_label[:int(g_size * 4 / 5)]

    return Train_data, Train_label, Test_data, Test_label

def read_satellite():
    filename = 'satellite.csv'
    ratio = 0.8
    num_feature = 36

    with open(filename) as f:
        reader = csv.reader(f)
        cont = list(reader)
    X_l = []
    y_l = []
    for i in cont:
        X_l.append(i[1:(num_feature+1)])
        y_l.append(i[0])
    X = np.array(X_l, dtype = float)
    y = np.array(y_l, dtype = int)

    # standarize and normalize the data
    y = y*2 - 1
    X = preprocessing.scale(X)

    # shuffle
    X,y = shuffle(X, y, random_state = 10086)
    size = len(y)
    X_train = X[:int(size*ratio)]
    y_train = y[:int(size*ratio)]
    X_test = X[int(size*ratio):]
    y_test = y[int(size*ratio):]

    return X_train, y_train, X_test, y_test


def read_2D():
    Train_data = np.load("Train_data_1.npy")
    Train_label = np.load("Train_label_1.npy")
    Train_label = 2 * Train_label - 1

    return Train_data, Train_label

def data_more(element, array):
    true_table = (array == element)

    size = np.size(element)

    sum_table = np.sum(true_table, axis=1)

    k = np.sum(sum_table==size)
    index = np.argwhere((sum_table==size))
    if (k>1):
        return True
    else:
        return False

def data_clean(Train_data, Train_label):
    size = np.size(Train_data, axis=1)
    for i in Train_data:
        true_table = (i == Train_data)
        sum_table = np.sum(true_table, axis=1)
        same_idx = np.argwhere((sum_table == size)).reshape((-1))

        Train_data = np.delete(Train_data, same_idx[1:], 0)
        Train_label = np.delete(Train_label, same_idx[1:])

    return Train_data, Train_label