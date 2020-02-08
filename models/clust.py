import pickle
from copy import deepcopy
import numpy as np

def distance(v1, v2):
    d = v1 - v2
    return np.dot(d, d)


def define_centers(dataset, n):
    centers = [dataset[0]]
    for i in range(1, n):
        dist_list = []
        for ranking in dataset:
            dist = [distance(ranking, center) for center in centers]
            min_dist = min(dist)
            dist_list.append(min_dist)
        centers.append(dataset[dist_list.index(max(dist_list))])
    return centers


def same(prev, pres):
    if max([distance(i, k) for i, k in zip(prev, pres)]) < 0.0001:
        return True
    else:
        return False


def clusterization(dataset, n_clusters):
    labels = []
    centroids = define_centers(dataset, n_clusters)
    prev_centroids = [[0] * len(centroids[0]) for i in range(len(centroids))]

    while not same(prev_centroids, centroids):
        labels = []
        prev_centroids = deepcopy(centroids)
        for ranking in dataset:
            dists = [distance(ranking, center) for center in centroids]
            idx = dists.index(min(dists))
            labels.append(idx)
        for i in range(n_clusters):
            num = 0
            temp = np.zeros(dataset[0].shape)
            for k, num_label in enumerate(labels):
                if num_label == i:
                    temp = temp + dataset[k]
                    num += 1
            centroids[i] = temp / num
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(dataset[idx])

    return clusters, centroids, labels

# pred = list()
# for i in range(len(y_pred)):
#     pred.append(np.argmax(y_pred[i]))
#
# test = list()
#
# for i in range(len(y_test)):
#     test.append(np.argmax(y_test[i]))
#
# from sklearn.metrics import accuracy_score
#
# a = accuracy_score(pred, test)
# print('Accuracy is:', a * 100)

