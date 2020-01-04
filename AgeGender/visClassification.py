import cv2
from config import ageGenderConfig as config
from config import ageGenderDeploy as deploy
from computervision.preprocessing import ImageToArrayPreprocessor
from computervision.preprocessing import SimplePreprocessor
from computervision.preprocessing import MeanPreprocessor
from computervision.utils import AgeGenderHelper
import numpy as np
import mxnet as mx
import argparse
import pickle
import imutils
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--sample-size", type=int, default=10)
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
genderModel = mx.model.FeedForward(ctx=[mx.gpu(0)], symbol = genderModel.symbol, arg_params = genderModel.arg_params, aux_params = genderModel.aux_params)

sp = SimplePreprocessor(width=227, height=227, inter=cv2.INTER_CUBIC)
ageMP = MeanPreprocessor(ageMeans["R"], ageMeans["G"], ageMeans["B"])
genderMP = MeanPreprocessor(genderMeans["R"], genderMeans["G"], genderMeans["B"])
iap = ImageToArrayPreprocessor()

rows = open(config.TEST_MX_LIST).read().strip().split("\n")
rows = np.random.choice(rows, size=args["sample_size"])

for row in rows:

    (_, gtLabel, imagePath) = row.strip().split("\t")
    image = cv2.imread(imagePath)

    ageImage = iap.preprocess(ageMP.preprocess(sp.preprocess(image)))
    genderImage = iap.preprocess(genderMP.preprocess(sp.preprocess(image)))
    ageImage = np.expand_dims(ageImage, axis=0)
    genderImage = np.expand_dims(genderImage, axis=0)

    agePreds = ageModel.predict(ageImage)[0]
    genderPreds = genderModel.predict(genderImage)[0]

    ageIdxs = np.argsort(agePreds)[::-1]
    genderIdxs = np.argsort(genderPreds)[::-1]

    ageCanvas = AgeGenderHelper.visualizeAge(agePreds, ageLE)

    genderCanvas = AgeGenderHelper.visualizeGender(genderPreds, genderLE)
    image = imutils.resize(image, width=400)

    gtLabel = ageLE.inverse_transform(int(gtLabel))
    text = "Actual: {}-{}".format(*gtLabel.split("_"))
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
    0.7, (0, 0, 255), 3)

    cv2.imshow("Image", image)
    cv2.imshow("Age Probabilities", ageCanvas)
    cv2.imshow("Gender Probabilities", genderCanvas)
    cv2.waitKey(0)