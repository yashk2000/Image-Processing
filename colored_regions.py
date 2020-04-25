from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import numpy as np
import argparse
import cv2

def segment_colorfulness(image, mask):
	(B, G, R) = cv2.split(image.astype("float"))
	R = np.ma.masked_array(R, mask=mask)
	G = np.ma.masked_array(B, mask=mask)
	B = np.ma.masked_array(B, mask=mask)

	rg = np.absolute(R - G)

	yb = np.absolute(0.5 * (R + G) - B)

	stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
	meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))

	return stdRoot + (0.3 * meanRoot)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
ap.add_argument("-s", "--segments", type=int, default=100)
args = vars(ap.parse_args())

orig = cv2.imread(args["image"])
vis = np.zeros(orig.shape[:2], dtype="float")

image = io.imread(args["image"])
segments = slic(img_as_float(image), n_segments=args["segments"],
	slic_zero=True)

for v in np.unique(segments):
	mask = np.ones(image.shape[:2])
	mask[segments == v] = 0

	C = segment_colorfulness(orig, mask)
	vis[segments == v] = C

vis = rescale_intensity(vis, out_range=(0, 255)).astype("uint8")

alpha = 0.6
overlay = np.dstack([vis] * 3)
output = orig.copy()
cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

cv2.imshow("Input", orig)
cv2.imshow("Visualization", vis)
cv2.imshow("Output", output)
cv2.waitKey(0)
