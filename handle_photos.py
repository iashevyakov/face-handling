# -*- coding: utf-8 -*-


import json
import operator
import pickle
import numpy as np

data_path = 'models/estimate.pickle'
features_path = './vects.pickle'

with open(data_path, "rb") as read_file:
    estimates = pickle.load(read_file)

with open(features_path, "rb") as read_file:
    features = pickle.load(read_file)

print(len(features['F']))
print(len(features['M']))

print(len(estimates['F']))
print(len(estimates['M']))

# feat_male = features['M']
# feat_female = features['F']
#
# features = {**feat_female, **feat_female}

# print(list(features.keys())[:5])
# features = {}
#
# for key in list(photo_vectors.keys()):
#     features[key.decode('utf-8')] = photo_vectors[key]
#
# with open('feats.pickle', "wb") as file:
#     pickle.dump(features, file)

female_estimates = sorted(estimates['F'].items(), key=operator.itemgetter(1), reverse=True)
print(female_estimates[:10])

male_estimates = sorted(estimates['M'].items(), key=operator.itemgetter(1), reverse=True)

female_X = []
male_X = []
female_y = []
male_y = []
#
n_classes = 3
male_part_count = int(len(male_estimates) / n_classes)
female_part_count = int(len(female_estimates) / n_classes)
male_rest = len(male_estimates) % n_classes
female_rest = len(female_estimates) % n_classes

print(male_rest)
print(female_rest)

for i in range(3):
    for photo, est in female_estimates[i * female_part_count: (i + 1) * female_part_count]:
        female_X.append(features['F'][photo])
        female_y.append(i)

    for photo, est in male_estimates[i * male_part_count: (i + 1) * male_part_count]:
        male_X.append(features['M'][photo])
        male_y.append(i)
if female_rest != 0:
    for i in range(n_classes * female_part_count, n_classes * female_part_count + female_rest):
        female_X.append(features['F'][female_estimates[i][0]])
        female_y.append(n_classes - 1)
if male_rest != 0:
    for i in range(n_classes * male_part_count, n_classes * male_part_count + male_rest):
        male_X.append(features['M'][male_estimates[i][0]])
        male_y.append(n_classes - 1)

print(len(female_X))
print(len(female_y))
print(len(male_X))
print(len(male_y))

female_X = np.array(female_X)
male_X = np.array(male_X)
female_y = np.array(female_y)
male_y = np.array(male_y)

percent_cls = {'F': (female_X, female_y), 'M': (male_X, male_y)}

with open('models/percls.pickle', 'wb') as f:
    pickle.dump(percent_cls, f)
