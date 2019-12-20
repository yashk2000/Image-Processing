import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from computervision.nn.conv import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required = True)
ap.add_argument("-m", "--models", required = True)
ap.add_argument("-n", "--num-models", type = int, default = 5)
args = vars(ap.parse_args())

((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb =LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

aug = ImageDataGenerator(rotation_range = 10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip = True, fill_mode = "nearest")

for i in np.arange(0, args["num_models"]):
    print("[INFO] Training model {}/{}".format(i + 1, args["num_models"]))
    opt = SGD(lr = 0.01, decay = 0.01 / 40, momentum = 0.9, nesterov = True)
    model = MiniVGGNet.build(width = 32, height = 32, depth = 3, classes = 10)
    model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

    H = model.fit_generator(aug.flow(trainX, trainY, batch_size = 64), validation_data = (testX, testY), epochs = 40, steps_per_epoch = len(trainX) // 64, verbose = 1)

    p = [args["models"], "model_{}.model".format(i)]
    model.save(os.path.sep.join(p))

    predictions = model.predict(testX, batch_size = 64)
    report = classification_report(testY.argmax(axis = 1), target_names = labelNames)

    p = [args["output"], "model_{}.txt".format(i)]
    f = open(os.path.sep.join(p), "w")
    f.write(report)
    f.close()

    p = [args["output"], "model_{}.png".format(i)]
    
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_accuracy")
    plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(os.path.sep.join(p))
    plt.close()