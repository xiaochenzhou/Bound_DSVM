import numpy as np
from sklearn import svm
import sklearn
import Edge_node as Ed

def index_non_outlier(Upload_support_vector_, Outlier_plus, Outlier_minus):
    n_edge_node = len(Upload_support_vector_)
    print(n_edge_node)

    Non_outlier_list = []
    for i in range(n_edge_node):
        idx = np.ones(np.size(Upload_support_vector_[i], axis=0), dtype=np.int32)
        idx_plus = np.arange(Outlier_plus[i])
        idx[idx_plus] = 0
        idx_minus = np.arange(int(np.size(idx)/2), int(np.size(idx)/2)+Outlier_minus[i])
        idx[idx_minus] = 0
        Non_outlier_list.append(idx)
    Non_outlier_ = np.concatenate((Non_outlier_list[0], Non_outlier_list[1]), axis=0)

    Non_outlier_ = np.argwhere((Non_outlier_ == 1)).reshape((-1))
    return Non_outlier_

def counter_update(Collect_support_vector_, Collect_label, gamma, C, Non_outlier_):

    rbf_kernel = sklearn.metrics.pairwise.rbf_kernel(Collect_support_vector_, Collect_support_vector_, gamma=gamma)
    # label_matrix = np.dot(np.transpose(Collect_label), Collect_label)
    #
    # Q_matrix = np.multiply(rbf_kernel, label_matrix)

    counter_line = np.dot((C*Collect_label), rbf_kernel).reshape((-1))

    Non_outlier_counter = counter_line[Non_outlier_]

    Pseudo_margin = np.max(Non_outlier_counter) - np.min(Non_outlier_counter)

    return Pseudo_margin, counter_line

def central_training(collected_data, collected_label, C, gamma, kernel):
    central_model = svm.SVC(C=C, kernel = kernel, gamma = gamma)
    central_model.fit(collected_data, collected_label)

    global_support_vector = central_model.support_vectors_
    global_label = collected_label[central_model.support_]

    return central_model, global_support_vector, global_label

def training_loss(model):
    dual_coef = model.dual_coef_.reshape((-1))
    support_vector = model.support_vectors_

    kernel_M = sklearn.metrics.pairwise.rbf_kernel(support_vector, support_vector, model.gamma)

    loss = 0.5 * np.dot(dual_coef, np.dot(kernel_M, dual_coef.T)) - np.sum(np.abs(dual_coef))

    return loss

def upper_bound(Edge_data, Edge_label, C, gamma, Edge_node_n):
    Edge_upper_loss = np.zeros(Edge_node_n)
    for i in range(Edge_node_n):
        # lower bound of training loss for global model
        local_model = Ed.local_train(Edge_data[i], Edge_label[i], C, gamma, 'rbf')
        # upper bound of training loss for global model
        upper_model = Ed.local_train(Edge_data[i], Edge_label[i], C * Edge_node_n, gamma, 'rbf')
        Edge_upper_loss[i] = \
            (1 / ( Edge_node_n)) * training_loss(upper_model)


    return np.sum(Edge_upper_loss)
