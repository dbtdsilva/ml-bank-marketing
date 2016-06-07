import csv
import math
import random


# Returns the dataset from the given csv filename
def loadCsv(filename):
	data = list(csv.reader(open(filename, "rb")))
	for i in range(len(data)):
		data[i] = [float(x) for x in data[i]]
	return data


# Randomly splits the dataset into the training set and test set 
# according to the given ratio 
def splitDataset(dataset, ratio):
	trainset = []
	testset = list(dataset)
	size = int(len(dataset) * ratio)
	while len(trainset) < size:
		trainset.append(testset.pop(random.randrange(len(testset))))
	return [trainset, testset]


# Returns the mean value of a vector
def mean(vector):
	return sum(vector)/float(len(vector))

 
 # Returns the standard deviation of a vector
def stdev(vector):
	avg = mean(vector)
	var = sum([pow(x-avg,2) for x in vector])/float(len(vector)-1)
	return math.sqrt(var)


# Separates the given set by label values
def byClassDict(dataset):
	byClass = {}
	for i in range(len(dataset)):
		v = dataset[i]
		if (v[-1] not in byClass):
			byClass[v[-1]] = []
		byClass[v[-1]].append(v)
	return byClass


# Returns a dictionary by label values with  the
# tuples (mean, standard deviation) for each attribute
def statisticsByClass(dataset):
	byClass = byClassDict(dataset)
	stats = {}
	for key, value in byClass.iteritems():
		stats[key] = [(mean(v), stdev(v)) for v in zip(*value)][:-1]
	return stats


# Gaussian formula:
# http://scikit-learn.org/stable/modules/naive_bayes.html
def getProbability(val, mean, stdev):
	exp = math.exp(-(math.pow(val-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exp


# Returns the probability for each possible label for the given attribute vector
def getLabelsProbabilities(stats, vector):
	probs = {}
	for label, labelStats in stats.iteritems():
		probs[label] = 0
		for i in range(len(labelStats)):
			mean, stdev = labelStats[i]
			probs[label] += math.log10(getProbability(vector[i], mean, stdev))
	return probs
			

# Returns the best label for the given attribute vector
def predict(stats, vector):
	probs = getLabelsProbabilities(stats, vector)
	bestLabel, bestProb = None, -1
	for label, prob in probs.iteritems():
		if bestLabel is None or prob > bestProb:
			bestProb = prob
			bestLabel = label
	return bestLabel
 

# Returns the best label for each attribute vector of the given set
def getPredictions(stats, vectorset):
	preds = []
	for i in range(len(vectorset)):
		preds.append(predict(stats, vectorset[i]))
	return preds


# Returns the overall accuracy of the classifier
def getAccuracy(testset, preds):
	c = 0
	for i in range(len(testset)):
		if testset[i][-1] == preds[i]:
			c += 1
	return c/float(len(testset))


# Returns the positive accuracy of the classifier
def getOneAccuracy(testset, preds):
	c = 0
	ones = 0
	for i in range(len(testset)):
		if testset[i][-1] == 1:
			ones += 1
			if preds[i] == 1:
				c += 1
	return c/float(ones)

def getPrecision(testset, preds):
	c = 0
	ones = 0
	for i in range(len(testset)):
		if preds[i] == 1:
			c += 1
			if testset[i][-1] == 1:
				ones += 1
			
	return ones/float(c)

def getFScore(testset, preds):
	recall = getOneAccuracy(testset, preds)
	precision = getPrecision(testset, preds)
	return (2 * precision * recall) / (recall + precision)


data = loadCsv('../data/bank-fixed-full.csv')

# Age Categorizer
ageUnit = 5
# Contact time Categorizer
timeUnit = 3

for i in range(0, len(data)):
	# Data Binning / Discretization

	# age -> round 5
	data[i][0] = round(data[i][0] / ageUnit) # * ageUnit
	# contact time -> 3 min intervals
	data[i][10] = round(round(data[i][10]/60) / timeUnit) # * timeUnit

	# euribor3m -> 1 decimal place
	#data[i][18] = round(data[i][18]*10) / 10

	# contacts during campaign -> round 5
	data[i][11] = round(data[i][11] / 5)

	# invert the previously inverted labels
	if data[i][20] == 1:
		data[i][20] = 0
	else:
		data[i][20] = 1

	# Remove noise attributes
	tmp = data[i]
	del tmp[14]
	del tmp[13]
	del tmp[12]
	del tmp[4]
	data[i] = tmp


train, test = splitDataset(data, 0.8)
stats = statisticsByClass(train)
preds = getPredictions(stats, test)

print 'Accuracy: ', getAccuracy(test, preds)*100, '%'
print 'Recall: ', getOneAccuracy(test, preds)*100, '%'
print 'Precision: ', getPrecision(test, preds)*100, '%'
print 'F-Score: ', getFScore(test, preds)*100, '%'
