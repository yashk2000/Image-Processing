import csv
import random
import math
import operator
 
def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for i in range(len(dataset)-1):
	        for j in range(4):
	            dataset[i][j] = float(dataset[i][j])
	        if random.random() < split:
	            trainingSet.append(dataset[i])
	        else:
	            testSet.append(dataset[i])
 
 
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for i in range(length):
		distance += pow((instance1[i] - instance2[i]), 2)
	return math.sqrt(distance)
 
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for i in range(k):
		neighbors.append(distances[i][0])
	return neighbors
 
def getResponse(neighbors):
	classVotes = {}
	for i in range(len(neighbors)):
		response = neighbors[i][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
 
def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
def main():
	trainingSet=[]
	testSet=[]
	split = 0.67
	loadDataset('iris.data', split, trainingSet, testSet)
	print("Train set: " + repr(len(trainingSet)))
	print("Test set: " + repr(len(testSet)))
	predictions=[]
	k = 3
	for i in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[i], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[i][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	
main()
