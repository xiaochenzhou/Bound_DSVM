import csv
import numpy as np
import sklearn as sl
from sklearn import svm, datasets
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
# for i in range ()
    filename = 'satellite.csv'
    if (filename == 'satellite.csv'):
        C = 12
        ratio = 0.8
        num_feature = 36
        k = 'rbf'
        ga = 1/num_feature
    elif (filename == 'pima.csv'):
        C = 1
        ratio = 0.8
        num_feature = 8
        k = 'rbf'
        ga = 1/num_feature
    else:
        raise('file_name_error')
    
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


    svm_sa = svm.SVC(C = C, kernel = k, gamma = ga)
    svm_sa.fit(X_train, y_train)
    acc = svm_sa.score(X_test, y_test)
    num_sv = svm_sa.n_support_
    y_test_p = svm_sa.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_test_p)
    print (acc)
    print (num_sv)
    print (conf_matrix)