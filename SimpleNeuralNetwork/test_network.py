from __future__ import print_function
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to output model file")
ap.add_argument("-t", "--test-images", required=True,
	help="path to the directory of testing images")
ap.add_argument("-b", "--batch-size", type=int, default=32,
	help="size of mini-batches passed to network")
args = vars(ap.parse_args())

CLASSES = ["cat", "dog"]

print("[INFO] loading network architecture and weights...")
model = load_model(args["model"])
print("[INFO] testing on images in {}".format(args["test_images"]))

for imagePath in paths.list_images(args["test_images"]):
	print("[INFO] classifying {}".format(
		imagePath[imagePath.rfind("/") + 1:]))
	image = cv2.imread(imagePath)
	features = image_to_feature_vector(image) / 255.0
	features = np.array([features])

	probs = model.predict(features)[0]
	prediction = probs.argmax(axis=0)

	label = "{}: {:.2f}%".format(CLASSES[prediction],
		probs[prediction] * 100)
	cv2.putText(image, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 255, 0), 3)
	cv2.imshow("Image", image)
	cv2.waitKey(0)