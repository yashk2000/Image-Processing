from keras.utils import np_utils
import numpy as np
import h5py


class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, preprocessors=None, aug=None, binarize=True, classes=2):
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

        self.db = h5py.File(dbPath)
        self.numImages = self.db['labels'].shape[0]

    def generator(self, passes=np.inf):
        epochs = 0

        while epochs < passes:
            for i in np.arange(0, self.numImages, self.batchSize):
                images = self.db['images'][i: i + self.batchSize]
                labels = self.db['labels'][i: i + self.batchSize]

                if self.binarize:
                    lables = np_utils.to_categorical(labels, self.classes)

                if self.preprocessors is not None:
                    procImages = []

                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        procImages.append(image)

                    images = np.array(procImages)

                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images, labels, batch_size=self.batchSize))

                yield (images, labels)

            epochs += 1

        def close(self):
            self.db.close()
