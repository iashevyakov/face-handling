import pickle

from models.fit import fit_logistic_regression
from joblib import load, dump

with open('scorecls.pickle', 'rb') as f:
    gender = pickle.load(f)

X_female, y_female = gender['F']
X_male, y_male = gender['M']

clf_female = fit_logistic_regression(X_female, y_female)
clf_male = fit_logistic_regression(X_male, y_male)

models = {'F': clf_female, 'M': clf_male}

dump(models, '../model.joblib')
