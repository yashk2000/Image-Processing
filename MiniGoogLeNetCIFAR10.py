import matplotlib
matplotlib.use("Agg")

import os
import argparse
import numpy as np
from computervision.nn.conv import MiniGoogLeNet
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from keras import backend as K

NUM_EPOCHS = 70
INIT_LR = 5e-3


def poly_decay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    return alpha


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True,
    help = "path to output model")
ap.add_argument("-o", "--output", required = True,
    help = "path to output directory (logs, plots, etc...")
args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 ")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

mean = np.mean(trainX, axis = 0)
trainX -= mean
testX -= mean

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

aug = ImageDataGenerator(
    width_shift_range = 0.1, horizontal_flip = True,
    fill_mode = "nearest"
)

figPath = os.path.sep.join([args["output"], f"{os.getpid()}.png"])
jsonPath = os.path.sep.join([args["output"], f"{os.getpid()}.json"])

print("[INFO] compiling ")
optimizer = SGD(lr = INIT_LR, momentum = 0.9)
model = MiniGoogLeNet.build(width = 32, height = 32, depth = 3, classes = 10)
model.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"])

print("[INFO] training")
model.fit_generator(
    aug.flow(trainX, trainY, batch_size = 64),
    validation_data = (testX, testY),
    steps_per_epoch = len(trainX) // 64,
    epochs = NUM_EPOCHS, verbose = 1
)

print("[INFO] serializing network...")
model.save(args["model"])
