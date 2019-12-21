import numpy as np
import cv2

class CropPreprocessor:
    def __init__(self, width, height, horiz = True, inter = cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter
    
    def preprocess(self, image):
        crops = []

        (h, w) = image.shape[:2]
        coords = [
            [0, 0, self.width, self.height],
            [w - self.width, 0, w, self.height],
            [w - self.width, h - self.height, w, h],
            [0, h - self.height, self.width, h]
        ]

        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (h - self.height))
        coords.append([dW, dH, w - dW, h - dH])

        for (startX, startY, endX, endY) in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.width, self.height), interpolation = self.inter)
            crops.append(crop)

        if self.horiz:
            mirros = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirros)

        return np.array(crops)