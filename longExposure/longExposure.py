import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True)
ap.add_argument("-o", "--output", required=True)
args = vars(ap.parse_args())

(Red, Green, Blue) = (None, None, None)
totalFrames = 0

vid = cv2.VideoCapture(args["video"])
print("[INFO] Calculating frame averages")

while True:
	(grabbed, frame) = vid.read()

	if not grabbed:
		break

	(B, G, R) = cv2.split(frame.astype("float"))

	if Red is None:
		Red = R
		Blue = B
		Green = G

	else:
		Red = ((totalFrames * Red) + (1 * R)) / (totalFrames + 1.0)
		Green = ((totalFrames * Green) + (1 * G)) / (totalFrames + 1.0)
		Blue = ((totalFrames * Blue) + (1 * B)) / (totalFrames + 1.0)

	totalFrames += 1

avg = cv2.merge([Blue, Green, Red]).astype("uint8")
cv2.imwrite(args["output"], avg)

vid.release()
