import cv2
from config import ageGenderDeploy as deploy
from computervision.preprocessing import ImageToArrayPreprocessor
from computervision.preprocessing import SimplePreprocessor
from computervision.preprocessing import MeanPreprocessor
from computervision.preprocessing import CropPreprocessor
from computervision.utils import AgeGenderHelper
from imutils.face_utils import FaceAligner
from imutils import face_utils
from imutils import paths
import numpy as np
import mxnet as mx
import argparse
import pickle
import imutils
import json
import dlib
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
args = vars(ap.parse_args())

print("[INFO] loading label encoders and mean files")
ageLE = pickle.loads(open(deploy.AGE_LABEL_ENCODER, "rb").read())
genderLE = pickle.loads(open(deploy.GENDER_LABEL_ENCODER, "rb").read())
ageMeans = json.loads(open(deploy.AGE_MEANS).read())
genderMeans = json.loads(open(deploy.GENDER_MEANS).read())

print("[INFO] loading models")
agePath = os.path.sep.join([deploy.AGE_NETWORK_PATH, deploy.AGE_PREFIX])
genderPath = os.path.sep.join([deploy.GENDER_NETWORK_PATH, deploy.GENDER_PREFIX])
ageModel = mx.model.FeedForward.load(agePath, deploy.AGE_EPOCH)
genderModel = mx.model.FeedForward.load(genderPath, deploy.GENDER_EPOCH)

print("[INFO] compiling models")
ageModel = mx.model.FeedForward(ctx = [mx.gpu(0)], symbol = ageModel.symbol, arg_params = ageModel.arg_params, aux_params = ageModel.aux_params)
genderModel = mx.model.FeedForward(ctx = [mx.gpu(0)], symbol = genderModel.symbol, arg_params = genderModel.arg_params, aux_params = genderModel.aux_params)

sp = SimplePreprocessor(width = 256, height = 256, inter = cv2.INTER_CUBIC)
cp = CropPreprocessor(width = 227, height = 227, horiz = True)
ageMP = MeanPreprocessor(ageMeans["R"], ageMeans["G"], ageMeans["B"])
genderMP = MeanPreprocessor(genderMeans["R"], genderMeans["G"], genderMeans["B"])
iap = ImageToArrayPreprocessor(dataFormat = "channels_first")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(deploy.DLIB_LANDMARK_PATH)
fa = FaceAligner(predictor)

imagePaths = [args["image"]]

if os.path.isdir(args["image"]):
    imagePaths = sorted(list(paths.list_files(args["image"])))

for imagePath in imagePaths:

    print("[INFO] processing {}".format(imagePath))
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for rect in rects:
        shape = predictor(gray, rect)
        face = fa.align(image, gray, rect)

        face = sp.preprocess(face)
        patches = cp.preprocess(face)

        agePatches = np.zeros((patches.shape[0], 3, 227, 227), dtype = "float")
        genderPatches = np.zeros((patches.shape[0], 3, 227, 227), dtype = "float")

        for j in np.arange(0, patches.shape[0]):
            agePatch = ageMP.preprocess(patches[j])
            genderPatch = genderMP.preprocess(patches[j])
            agePatch = iap.preprocess(agePatch)
            genderPatch = iap.preprocess(genderPatch)

            agePatches[j] = agePatch
            genderPatches[j] = genderPatch

        agePreds = ageModel.predict(agePatches)
        genderPreds = genderModel.predict(genderPatches)

        agePreds = agePreds.mean(axis=0)
        genderPreds = genderPreds.mean(axis=0)

        ageCanvas = AgeGenderHelper.visualizeAge(agePreds, ageLE)
        genderCanvas = AgeGenderHelper.visualizeGender(genderPreds,
        genderLE)

        clone = image.copy()
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Input", clone)
        cv2.imshow("Face", face)
        cv2.imshow("Age Probabilities", ageCanvas)
        cv2.imshow("Gender Probabilities", genderCanvas)
        cv2.waitKey(0)

