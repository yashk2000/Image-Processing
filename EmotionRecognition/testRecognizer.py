from config import EmotionConfig as config
from computervision.preprocessing import ImageToArrayPreprocessor
from computervision.io import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str)
args = vars(ap.parse_args())

testAug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()

testGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE, aug = testAug, preprocessors = [iap], classes = config.NUM_CLASSES)

print("[INFO] loading {}".format(args["model"]))
model = load_model(args["model"])

(loss, acc) = model.evaluate_generator( testGen.generator(), steps=testGen.numImages // config.BATCH_SIZE, max_queue_size = config.BATCH_SIZE * 2)
print("[INFO] accuracy: {:.2f}".format(acc * 100))

testGen.close()