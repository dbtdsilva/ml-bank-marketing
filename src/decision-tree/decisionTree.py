import csv
import random


class decisionnode:
    def __init__(self, col = -1, value = None, results = None, tb = None, fb = None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb  # true branch
        self.fb = fb  # false branch


# Divides a set on a specific column
def divideset(rows, column, v):
    func = lambda row: row[column] == v
    if isinstance(v, int) or isinstance(v, float):
        func = lambda row: row[column] >= v

    # Divide the rows into two sets and return them
    set1 = [row for row in rows if func(row)]
    set2 = [row for row in rows if not func(row)]
    return set1, set2

# Calculates the amount of unique possible results
# The value is stored in the last column of each row
def uniquecounts(rows):
    results = {}
    for row in rows:
        r = row[len(row) - 1]
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results


from math import log


# Entropy = SUM ( p(x) log(p(x)) )
def entropy(rows):
    log2 = lambda x: log(x) / log(2)
    results = uniquecounts(rows) # Calculate unique possible values
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent -= p * log2(p)
    return ent

# Function to classify the test set values
def classify(observation, tree):
    if tree.results != None:
        return tree.results
    else:
        v = observation[tree.col]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        return classify(observation, branch)

# Function to build the decision tree
def buildtree(rows, scoref = entropy):
    if len(rows) == 0:
        return decisionnode()
    current_score = scoref(rows)

    best_gain = 0.0  # Best Information Gain
    best_criteria = None  # Best Criteria = (Column, Value)
    best_sets = None  # Best Sets = (Set1, Set2)

    for col in range(0, len(rows[0]) - 1):
        column_values = {}  # Different values in this column
        for row in rows:
            column_values[row[col]] = 1

        # Divide the rows for each value
        for value in column_values.keys():
            (set1, set2) = divideset(rows, col, value)

            p = float(len(set1)) / len(rows)
            # Information gain
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)

            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)

    # Create the sub branches
    if best_gain > 0:
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        return decisionnode(col = best_criteria[0], value = best_criteria[1],
                            tb = trueBranch, fb = falseBranch)
    else:
        return decisionnode(results = uniquecounts(rows))


# Gets the data from the .csv file
def loadCsv(filename):
    data = list(csv.reader(open(filename, "rb")))
    for i in range(len(data)):
        data[i] = [float(x) for x in data[i]]
    return data


# Splits the dataset into a
# Training Set and
# Test Set
def splitDataset(dataset, ratio):
    trainset = []
    testset = list(dataset)
    size = int(len(dataset) * ratio)
    while len(trainset) < size:
        trainset.append(testset.pop(random.randrange(len(testset))))
    return [trainset, testset]


data = loadCsv('../data/bank-fixed-full.csv')

# Age Categorizer
ageUnit = 5
# Contact time Categorizer
timeUnit = 3

for i in range(0, len(data)):
    # Data Binning / Discretization

    # age -> round 5
    data[i][0] = round(data[i][0] / ageUnit)  # * ageUnit
    # contact time -> 3 min intervals
    data[i][10] = round(round(data[i][10] / 60) / timeUnit)  # * timeUnit

    # euribor3m -> 1 decimal place
    # data[i][18] = round(data[i][18]*10) / 10

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


# Returns the overall accuracy of the classifier
def getAccuracy(testset, preds):
    c = 0
    for i in range(len(testset)):
        if testset[i][-1] == preds[i]:
            c += 1
    return c / float(len(testset))


# Returns the positive accuracy of the classifier
def getOneAccuracy(testset, preds):
    c = 0
    ones = 0
    for i in range(len(testset)):
        if testset[i][-1] == 1:
            ones += 1
            if preds[i] == 1:
                c += 1
    return c / float(ones)


# Returns the precision of the classifier
def getPrecision(testset, preds):
    c = 0
    ones = 0
    for i in range(len(testset)):
        if preds[i] == 1:
            c += 1
            if testset[i][-1] == 1:
                ones += 1

    return ones / float(c)


# Returns the F-Measure/Score of the classifier
def getFScore(testset, preds):
    recall = getOneAccuracy(testset, preds)
    precision = getPrecision(testset, preds)
    return (2 * precision * recall) / (recall + precision)


train, test = splitDataset(data, 0.7)

tree = buildtree(train)
preds = []
for item in test:
    preds += [classify(item, tree).keys()[0]]

print 'Accuracy: ', getAccuracy(test, preds) * 100, '%'
print 'Recall: ', getOneAccuracy(test, preds) * 100, '%'
print 'Precision: ', getPrecision(test, preds) * 100, '%'
print 'F-Score: ', getFScore(test, preds) * 100, '%'

# print result
