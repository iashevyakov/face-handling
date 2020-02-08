import pickle
from copy import deepcopy
from random import shuffle

import cv2
import numpy as np
import openface
from sklearn.model_selection import KFold
from joblib import dump
from models.clfs import run_sequential, run_logistic_regression, run_svm, run_tree, run_forest, run_knn
from models.clust import clusterization

# from models.fit import fit_logistic_regression
from models.fit import *

dlibFacePredictor = '/home/ivan/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
networkModel = '/home/ivan/openface/models/openface/nn4.small2.v1.t7'
align = openface.AlignDlib(dlibFacePredictor)
net = openface.TorchNeuralNet(networkModel, 96)
models_path = '../model.joblib'


def preprocess(imgPath, clf):
    bgrImg = cv2.imread(imgPath)

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    bb = align.getAllFaceBoundingBoxes(rgbImg)

    class_names = ('красивый', 'обычный', 'некрасивый')

    if len(bb) > 1:
        raise Exception('Найдено больше одного лица!')
    elif len(bb) == 0:
        raise Exception('Не найдено ни одного лица!')
    else:
        bb = align.getAllFaceBoundingBoxes(rgbImg)[0]
        try:
            alignedFace = align.align(96, rgbImg, bb,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            rep = net.forward(alignedFace).reshape(1, -1)

            cls_name = class_names[clf.predict(rep)[0]]
            print(cls_name)
        except:
            raise Exception('Не удаётся извлечь признаки, выберите другую фотографию!')


with open('scorecls.pickle', 'rb') as f:
    gender = pickle.load(f)

X, y = gender['M']

clf = fit_logistic_regression(X, y)

img1 = '/home/ivan/faces/Kristina_Фомина_75814093_M_4.jpg'
img2 = '/home/ivan/face_handling/img/debcb730f56f9019fa7b2e46051e6ca8774ffa66.jpg'
img3 = '/home/ivan/face_handling/1.jpg'
img4 = '/home/ivan/face_handling/vEv6fMIFvo8.jpg'
img5 = '/home/ivan/face_handling/img/51.jpg'
img6 = '/home/ivan/face_handling/img/894eb22bfad6148ff88ac0e80ff0b331221d54f6.jpg'


preprocess(img6, clf)
