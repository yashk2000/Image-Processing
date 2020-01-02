from keras.callbacks import Callback
import os

class EpochCheckpoint(Callback):
	def __init__(self, outputPath, every=5, startAt = 0):
		super(Callback, self).__init__()

		self.outputPath = outputPath
		self.every = every
		self.intEpoch = startAt

	def on_epoch_end(self, epoch, logs={}):
		if (self.intEpoch + 1) % self.every == 0:
			p = os.path.sep.join([self.outputPath, "epoch_{}.hdf5".format(self.intEpoch + 1)])
			self.model.save(p, overwrite=True)

		self.intEpoch += 1
