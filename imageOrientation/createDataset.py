from imutils import paths
import numpy as np
import progressbar
import argparse
import imutils
import random
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True)
ap.add_argument("-o", "--output", required = True)
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))[:10000]
random.shuffle(imagePaths)

angles = {}
widgets = ["Building dataset:", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval = len(imagePaths), widgets = widgets).start()

for (i, imagePath) in enumerate(imagePaths):
    angle = np.random.choice([0, 90, 180,  270])
    image = cv2.imread(imagePath)

    if image is None:
        continue

    image = imutils.rotate_bound(image, angle)
    base = os.path.sep.join([args["output"], str(angle)])

    if not os.path.exists(base):
        os.makedirs(base)

    ext = imagePath[imagePath.rfind("."):]
    outputPath = [base, "image_{}{}".format(str(angles.get(angle, 0)).zfill(5), ext)]
    outputPath = os.path.sep.join(outputPath)

    cv2.imwrite(outputPath, image)

    c = angles.get(angle, 0)
    angles[angle] = c + 1
    pbar.update(i)

    for angle in sorted(angles.keys()):
        print("[INFO] angle = {}: {:,}".format(angle, angles[angle]))
        