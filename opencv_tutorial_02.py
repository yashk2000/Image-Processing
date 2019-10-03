import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Image", image)
 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)

edged = cv2.Canny(gray, 30, 150)
cv2.imshow("Edged", edged)

thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresh", thresh)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()
 
for c in cnts:
	cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
	cv2.imshow("Contours", output)

maskThresh = thresh.copy()
maskThresh = cv2.erode(maskThresh, None, iterations=5)
cv2.imshow("Eroded", maskThresh)

maskDilate = thresh.copy()
maskDilate = cv2.dilate(maskDilate, None, iterations=5)
cv2.imshow("Dilated", maskDilate)

mask = thresh.copy()
op = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Op", op)

cv2.waitKey(0)