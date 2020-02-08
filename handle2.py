import operator
import pickle
import numpy as np

with open('models/estimate.pickle', "rb") as read_file:
    estimates = pickle.load(read_file)

with open('vects.pickle', "rb") as read_file:
    features = pickle.load(read_file)

print(len(features['F']))
print(len(features['M']))

print(len(estimates['F']))
print(len(estimates['M']))

female_estimates = sorted(estimates['F'].items(), key=operator.itemgetter(1), reverse=True)

male_estimates = sorted(estimates['M'].items(), key=operator.itemgetter(1), reverse=True)

female_X = []
male_X = []
female_y = []
male_y = []

start = 0.4
end = 0.7

for photo, est in female_estimates:
    female_X.append(features['F'][photo])
    if est >= 0.7:
        y = 0
    elif 0.4 <= est < 0.7:
        y = 1
    else:
        y = 2
    female_y.append(y)

for photo, est in male_estimates:
    male_X.append(features['M'][photo])
    if est >= end:
        y = 0
    elif start <= est < end:
        y = 1
    else:
        y = 2
    male_y.append(y)

print(len(female_X))
print(len(female_y))
print(len(male_X))
print(len(male_y))

female_X = np.array(female_X)
male_X = np.array(male_X)
female_y = np.array(female_y)
male_y = np.array(male_y)

score_cls = {'F': (female_X, female_y), 'M': (male_X, male_y)}

with open('models/scorecls.pickle', 'wb') as f:
    pickle.dump(score_cls, f)
