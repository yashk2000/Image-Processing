import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to the image file")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

for angle in np.arange(0, 360, 15):
    rotated = imutils.rotate(image, angle)
    cv2.imshow("Rotated (not proper)", rotated)
    cv2.waitKey(0)

for angle in np.arange(0, 360, 15):
    rotated = imutils.rotate_bound(image, angle)
    cv2.imshow("Rotated (correct)", rotated)
    cv2.waitKey(0)
    