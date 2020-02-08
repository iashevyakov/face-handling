import json
import pickle

with open('vectors2.pickle', 'rb') as f:
    features = pickle.load(f)

print(features.keys())
print(len(features['F']))
print(len(features['M']))
# c = 0
#

exc_female = []
for key in features['F']:
    if features['F'][key] is None:
        exc_female.append(key)

exc_male = []
for key in features['M']:
    if features['M'][key] is None:
        exc_male.append(key)

with open('rating_export.json', "r") as read_file:
    estimates = json.load(read_file, encoding='utf-8')

for key in exc_female:
    del estimates['F'][key]

for key in exc_male:
    del estimates['M'][key]

print(len(estimates['F']))
print(len(estimates['M']))

#
# print(len(exc_female))
# print(len(exc_male))
#
# for key in exc_female:
#     del features['F'][key]
#
# for key in exc_male:
#     del features['M'][key]
#
with open('models/estimate.pickle', 'wb') as f:
    pickle.dump(estimates, f)
