from copy import deepcopy
from random import shuffle

import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# with open('../female.pickle', 'rb') as f:
#     gender = pickle.load(f)
#
# X = gender[0]
#
# n_clusters = 3
# #
# clusters, centroids, labels = clusterization(X, n_clusters)
#
# classes = deepcopy(labels)
#
# part = int(2.2 * len(classes) / 3)
#
# cl1 = deepcopy(classes[:part])
# cl2 = deepcopy(classes[part:])
#
# shuffle(cl2)
#
# cl1.extend(cl2)
#
# y = np.array(cl1)


def fit_sequential(X, y):
    y = to_categorical(y)

    model = Sequential()
    # model.add(Dense(128, activation='relu', input_dim=128))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(150, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(9, activation='relu'))
    # model.add(Dense(5, activation='softmax'))
    model.add(Dense(120, activation='relu', input_dim=128))
    # model.add(Dense(80, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X, y, epochs=100, batch_size=64, verbose=0)
    # print('Accuracy on test data: {}'.format(acc))
    return model


def fit_logistic_regression(X, y):
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

    clf.fit(X, y)

    return clf


def fit_svm(X, y):
    clf = svm.SVC(gamma='scale')

    clf.fit(X, y)

    return clf


def fit_tree(X, y):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y)
    return clf


def fit_knn(X, y):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X, y)
    return clf


def fit_forest(X, y):
    clf = RandomForestClassifier(random_state=0, n_estimators=100)
    clf.fit(X, y)
    return clf
