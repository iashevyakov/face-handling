# -*- coding: utf-8 -*-
import json
import logging
import operator
import pickle
import time

import requests

import os
import time
import cv2
import itertools
import numpy as np

np.set_printoptions(precision=2)

import openface

dlibFacePredictor = '/root/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
networkModel = '/root/openface/models/openface/nn4.small2.v1.t7'
align = openface.AlignDlib(dlibFacePredictor)
net = openface.TorchNeuralNet(networkModel, 96)

root_path = '/home/ubuntu/project/kafis/backend/kafis/media/faces'
# root_path = './img'

female_vectors = {}

male_vectors = {}

pickle_data_path = './vectors2.pickle'


def getRepr(imgPath):
    try:
        bgrImg = cv2.imread(imgPath)

        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

        bb = align.getLargestFaceBoundingBox(rgbImg, skipMulti=True)
        rep = None
        if bb:
            alignedFace = align.align(96, rgbImg, bb,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            rep = net.forward(alignedFace)

    except:
        rep = None
        print('except')

    return rep


logging.basicConfig(filename="state5.log", level=logging.INFO)
log = logging.getLogger("ex")

with open('rating_export.json', "r") as read_file:
    estimates = json.load(read_file, encoding='utf-8')

# estimates = {
#     'F': {'4fab17def36254fe66f28a9e0ae23453b77e9bb0.jpg': 0.7, 'debcb730f56f9019fa7b2e46051e6ca8774ffa66.jpg': 0.5},
#     'M': {'894eb22bfad6148ff88ac0e80ff0b331221d54f6.jpg': 0.6}}

for idx, felame_photo in enumerate(estimates['F']):
    female_vectors[felame_photo] = getRepr(os.path.join(root_path, felame_photo))
    log.info("Handles {} female".format(idx))

for idx, male_photo in enumerate(estimates['M']):
    male_vectors[male_photo] = getRepr(os.path.join(root_path, male_photo))
    log.info("Handles {} male".format(idx))

photo_vectors = {'F': female_vectors, 'M': male_vectors}

log.info("Start pickling")
with open(pickle_data_path, 'wb') as f:
    pickle.dump(photo_vectors, f)
