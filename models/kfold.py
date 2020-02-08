import pickle
from copy import deepcopy
from random import shuffle

import numpy as np
from sklearn.model_selection import KFold
from joblib import dump
from models.clfs import run_sequential, run_logistic_regression, run_svm, run_tree, run_forest, run_knn, run_mlp
from joblib import load, dump
from models.fit import fit_logistic_regression

with open('feats.pickle', 'rb') as f:
    gender = pickle.load(f)

X_male, y_male = gender['M']
X_female, y_female = gender['F']

clf_female = fit_logistic_regression(X_female, y_female)
clf_male = fit_logistic_regression(X_male, y_male)

models = {'F': clf_female, 'M': clf_male}

dump(models, 'model.joblib')

# kf = KFold(n_splits=10, random_state=None, shuffle=True)
# iter_acc = []
#
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     acc = round(run_logistic_regression(X_train, X_test, y_train, y_test), 2)
#
#     iter_acc.append(acc)
#     print('Accuracy on test data {}'.format(acc))
#
# print('mean {}'.format(np.mean(iter_acc)))
