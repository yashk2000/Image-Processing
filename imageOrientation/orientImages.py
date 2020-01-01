from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import argparse
import pickle
import imutils
import h5py
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True)
ap.add_argument("-i", "--image", required=True)
ap.add_argument("-m", "--model", required=True)
args = vars(ap.parse_args())

db = h5py.File(args["db"])
labelNames = [int(angle) for angle in db["label_names"][:]]
db.close()

print("[INFO] sampling image")
imagePath = args["image"]

print("[INFO] loading network")
vgg = VGG16(weights="imagenet", include_top=False)

print("[INFO] loading model")
model = pickle.loads(open(args["model"], "rb").read())

orig = cv2.imread(imagePath)

image = load_img(imagePath, target_size=(224, 224))
image = img_to_array(image)

image = np.expand_dims(image, axis=0)
image = imagenet_utils.preprocess_input(image)

features = vgg.predict(image)
features = features.reshape((features.shape[0], 512 * 7 * 7))

angle = model.predict(features)
angle = labelNames[angle[0]]

rotated = imutils.rotate_bound(orig, 360 - angle)

cv2.imshow("Original", orig)
cv2.imshow("Corrected", rotated)
cv2.waitKey(0)
