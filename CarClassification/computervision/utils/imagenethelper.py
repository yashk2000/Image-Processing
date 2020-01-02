import numpy as np
import os

class ImageNetHelper:
	def __init__(self, config):
		self.config = config

		self.labelMappings = self.buildClassLabels()
		self.valBlacklist = self.buildBlackist()

	def buildClassLabels(self):
		rows = open(self.config.WORD_IDS).read().strip().split("\n")
		labelMappings = {}

		for row in rows:
			(wordID, label, hrLabel) = row.split(" ")

			labelMappings[wordID] = int(label) - 1

		return labelMappings

	def buildBlackist(self):
		rows = open(self.config.VAL_BLACKLIST).read()
		rows = set(rows.strip().split("\n"))

		return rows

	def buildTrainingSet(self):
		rows = open(self.config.TRAIN_LIST).read().strip()
		rows = rows.split("\n")
		paths = []
		labels = []

		for row in rows:
			(partialPath, imageNum) = row.strip().split(" ")

			path = os.path.sep.join([self.config.IMAGES_PATH, "train", "{}.JPEG".format(partialPath)])
			wordID = partialPath.split("/")[0]
			label = self.labelMappings[wordID]

			paths.append(path)
			labels.append(label)

		return (np.array(paths), np.array(labels))

	def buildValidationSet(self):
		paths = []
		labels = []

		valFilenames = open(self.config.VAL_LIST).read()
		valFilenames = valFilenames.strip().split("\n")

		valLabels = open(self.config.VAL_LABELS).read()
		valLabels = valLabels.strip().split("\n")

		for (row, label) in zip(valFilenames, valLabels):
			(partialPath, imageNum) = row.strip().split(" ")

			if imageNum in self.valBlacklist:
				continue

			path = os.path.sep.join([self.config.IMAGES_PATH, "val", "{}.JPEG".format(partialPath)])
			paths.append(path)
			labels.append(int(label) - 1)

		return (np.array(paths), np.array(labels))