# -*- coding: utf-8 -*-
import argparse

import cv2
import openface
from joblib import load
from PIL import Image

from cert import cert


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

dlibFacePredictor = '/home/ivan/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
networkModel = '/home/ivan/openface/models/openface/nn4.small2.v1.t7'
align = openface.AlignDlib(dlibFacePredictor)
net = openface.TorchNeuralNet(networkModel, 96)
models_path = 'model.joblib'

parser = argparse.ArgumentParser()

parser.add_argument('--gender', type=str, default='F')
parser.add_argument('--name', type=str, default='Кристина Фомина')
parser.add_argument('--img', type=str, default='/home/ivan/faces/Kristina_Фомина_75814093_M_4.jpg')

args = parser.parse_args()


def preprocess(imgPath, gender='F', name=None):
    bgrImg = cv2.imread(imgPath)

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    bb = align.getAllFaceBoundingBoxes(rgbImg)

    class_names = ('красивый', 'средней красоты', 'некрасивый')

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

            left = max(int(bb.left() - bb.width() * 0.5), 0)
            right = min(int(bb.right() + bb.width() * 0.5), int(rgbImg.shape[1]))
            top = max(int(bb.top() - bb.height() * 0.5), 0)
            bottom = min(int(bb.bottom() + bb.height() * 0.5), int(rgbImg.shape[0]))

            models = load(models_path)

            cls_name = class_names[models[gender].predict(rep)[0]]
            face = Image.fromarray(rgbImg[top:bottom, left:right])
            img = cert(name, face, name, cls_name)
            return img
        except Exception as e:
            raise Exception('Не удаётся извлечь признаки, выберите другую фотографию!')


img1 = '/home/ivan/faces/Kristina_Фомина_75814093_M_4.jpg'
img2 = '/home/ivan/faces/Ainura_Zhasenova_196927861_M_1.jpg'
# img2 = '/home/ivan/faces/img/yandex2.jpg'
# img3 = '/home/ivan/faces/img/grim.jpg'
img4 = '/home/ivan/faces/img/kravez.jpg'
# # try:
# #     preprocess(img1, gender='F')
# # except Exception as e:
# #     print(str(e))
#


try:
    img = preprocess(args.img, args.gender, args.name)

    img.show()
except Exception as e:
    print(str(e))
