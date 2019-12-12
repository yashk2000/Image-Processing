from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from computervision.preprocessing import ImageToArrayPreprocessor
from computervision.preprocessing import SimplePreprocessor
from computervision.datasets import SimpleDatasetLoader
from computervision.nn.conv import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True)
ap.add_argument("-m", "--model", required = True)
args = vars(ap.parse_args())

print("[INFO] Loading")
imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors = [sp, iap])
(data, labels) = sdl.load(imagePaths, verbose = 500)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] Compiling")
opt = SGD(lr = 0.005)
model = ShallowNet.build(width = 32, height = 32, depth = 3, classes = 3)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

print("[INFO] Training")
H = model.fit(trainX, trainY, validation_data = (testX, testY), batch_size = 32, epochs = 100, verbose = 1)

print("[INFO] serializing")
model.save(args["model"])

print("[INFO] Evaluating")
predictions = model.predict(testX, batch_size = 32)
print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1), target_names = ["cat", "dog", "panda"]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()