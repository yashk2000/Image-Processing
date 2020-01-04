from config import ageGenderConfig as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from computervision.utils import AgeGenderHelper
import numpy as np
import progressbar
import pickle
import json
import cv2

print("[INFO] building paths and labels")
agh = AgeGenderHelper(config)
(trainPaths, trainLabels) = agh.buildPathsAndLabels()

numVal = int(len(trainPaths) * config.NUM_VAL_IMAGES)
numTest = int(len(trainPaths) * config.NUM_TEST_IMAGES)

print("[INFO] encoding labels...")
le = LabelEncoder().fit(trainLabels)
trainLabels = le.transform(trainLabels)

print("[INFO] constructing validation data...")
split = train_test_split(trainPaths, trainLabels, test_size=numVal, stratify=trainLabels)
(trainPaths, valPaths, trainLabels, valLabels) = split

print("[INFO] constructing testing data...")
split = train_test_split(trainPaths, trainLabels, test_size=numTest, stratify=trainLabels)
(trainPaths, testPaths, trainLabels, testLabels) = split

datasets = [
    ("train", trainPaths, trainLabels, config.TRAIN_MX_LIST),
    ("val", valPaths, valLabels, config.VAL_MX_LIST),
    ("test", testPaths, testLabels, config.TEST_MX_LIST)
]

(R, G, B) = ([], [], [])

for (dType, paths, labels, outputPath) in datasets:

    print("[INFO] building {}".format(outputPath))
    f = open(outputPath, "w")
    widgets = ["Building List: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    for (i, (path, label)) in enumerate(zip(paths, labels)):

        if dType == "train":
            image = cv2.imread(path)
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        row = "\t".join([str(i), str(label), path])
        f.write("{}\n".format(row))
        pbar.update(i)

    pbar.finish()
    f.close()

print("[INFO] serializing means")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()

print("[INFO] serializing label encoder")
f = open(config.LABEL_ENCODER_PATH, "wb")
f.write(pickle.dumps(le))
f.close()
