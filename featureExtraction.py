import os
import random
import argparse
import progressbar
import numpy as np
from imutils import paths
from computervision.io import HDF5DatasetWriter
from sklearn.preprocessing import LabelEncoder
from keras.applications import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True)
ap.add_argument("-o", "--output", required = True)
ap.add_argument("-b", "--batch-size", type = int, default = 32)
ap.add_argument("-s", "--buffer-size", type = int, default = 1000)
args = vars(ap.parse_args())

print("[INFO] loading")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

labels = [path.split(os.path.sep)[-2] for path in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

print("[INFO] loading network...")
model = VGG16(weights = "imagenet", include_top = False)

dataset = HDF5DatasetWriter(dims = (len(imagePaths), 512 * 7 * 7), outputPath = args["output"], dataKey = "features", bufSize = args["buffer_size"])
dataset.storeClassLabels(le.classes_)


widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval = len(imagePaths), widgets = widgets).start()
batchSize = args["batch_size"]

for i in np.arange(0, len(imagePaths), batchSize):
    batchPaths = imagePaths[i:i + batchSize]
    batchLabels = labels[i:i + batchSize]
    batchImages = []

    for (j, imagePath) in enumerate(batchPaths):
        image = load_img(imagePath, target_size = (224, 244))
        image = img_to_array(image)
        image = np.expand_dims(image, axis = 0)
        image = imagenet_utils.preprocess_input(image)

        batchImages.append(image)

    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size = batchSize)

    features = features.reshape((features.shape[0], 512 * 7 * 7))

    dataset.add(features, batchLabels)
    pbar.update(i)

dataset.close()
pbar.finish()