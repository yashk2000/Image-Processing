import cv2

class MeanPreprocessor:

    def __init___(self, rMean, gMean, bMean):
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def preprocessor(self, image):
        (B, G, R) = cv2.split(image.astype("float32"))

        R -= self.rMean
        G -= self.gMean
        B -= self.bMean

        return cv2.merge([B, G, R])
        