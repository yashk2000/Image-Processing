import matplotlib
matplotlib.use("Agg")

from config import EmotionConfig as config
from computervision.preprocessing import ImageToArrayPreprocessor
from computervision.callbacks import TrainingMonitor
from computervision.callbacks import EpochCheckpoint
from computervision.io import HDF5DatasetGenerator
from computervision.nn.conv import EmotionVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend as K
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required = True)
ap.add_argument("-m", "--model", type = str)
ap.add_argument("-s", "--start-epoch", type = int, default = 0)
args = vars(ap.parse_args())

trainAug = ImageDataGenerator(rotation_range = 10, zoom_range = 0.1, horizontal_flip = True, rescale = 1 / 255.0, fill_mode = "nearest")
valAug = ImageDataGenerator(rescale = 1 / 255.0)
iap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug = trainAug, preprocessors = [iap], classes = config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, aug = valAug, preprocessors = [iap], classes = config.NUM_CLASSES)

if args["model"] is None:
	print("[INFO] compiling")
	model = EmotionVGGNet.build(width = 48, height = 48, depth = 1, classes = config.NUM_CLASSES)
	opt = Adam(lr = 1e-3)
	model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

else:
	print("[INFO] loading {}".format(args["model"]))
	model = load_model(args["model"])

	print("[INFO] old lr: {}".format(K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr, 1e-5)
	print("[INFO] new lr: {}".format(K.get_value(model.optimizer.lr)))

figPath = os.path.sep.join([config.OUTPUT_PATH, "vggnet_emotion.png"])
callbacks = [
	EpochCheckpoint(args["checkpoints"], every=5, startAt = args["start_epoch"]),
]

model.fit_generator(
	trainGen.generator(),
	steps_per_epoch = trainGen.numImages // config.BATCH_SIZE,
	validation_data = valGen.generator(),
	validation_steps = valGen.numImages // config.BATCH_SIZE,
	epochs = 15,
	max_queue_size = config.BATCH_SIZE * 2,
	callbacks = callbacks,
	verbose = 1
)

trainGen.close()
valGen.close()