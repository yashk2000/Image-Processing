from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2 
import os

class MiniVGGNet:

    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = -1

        model.add(Conv2D(32, (3, 3), padding = "same", input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(32, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes)) 
        model.add(Activation("softmax"))

        return model    

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True)
ap.add_argument("-m", "--model", required = True)
args = vars(ap.parse_args())

data = []
labels = []

for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width = 28)
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)

data = np.array(data, dtype = "float") / 255.0
labels = np.array(labels)

le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

classTotals = labels.sum(axis = 0)
classWeight = classTotals.max() / classTotals

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.20, stratify = labels, random_state = 42)

print("[INFO] Compiling")
model = MiniVGGNet.build(width = 28, height = 28, depth = 1, classes = 2)
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

print("[INFO] Training")
H = model.fit(trainX, trainY, validation_data = (testX, testY), class_weight = classWeight, batch_size = 64, epochs = 15, verbose = 1)

print("[INFO] Evaluating network")
predictions = model.predict(testX, batch_size = 64)
print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1), target_names = le.classes_))

print("[INFO] Serializing")
model.save(args["model"])

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, 15), H.history["accuracy"], label = "accuracy")
plt.plot(np.arange(0, 15), H.history["val_accuracy"], label = "val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()