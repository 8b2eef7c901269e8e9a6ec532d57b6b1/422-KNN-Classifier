# CS422 - Data Analytics
# Jared McLean & Jared Schreiber
# 2/22/2017
"K-Nearest-Neighbors Classifier with test data"

import csv
import math
import operator
import copy
import sys

#offset values by a small amount to avoid division by 0
EPSI = sys.float_info.epsilon

#functions

#main function
def main():
    "main function"
    BUPA_NUM_ATTRIBUTES = 7
    CAR_NUM_ATTRIBUTES = 7

    BUPA_CLASS_INDEX = 6
    CAR_CLASS_INDEX = 6

    MIN_K = 15
    MAX_K = 15
    CAR_VALUES = [
        {'vhigh' : 4.0,
         'high' : 3.0,
         'med' : 2.0,
         'low' : 1.0},
        {'vhigh' : 4.0,
         'high' : 3.0,
         'med' : 2.0,
         'low' : 1.0},
        {'5more' : 5.0},
        {'more' : 6.0},
        {'small' : 1.0,
         'med' : 2.0,
         'big' : 3.0},
        {'low' : 1.0,
         'med' : 2.0,
         'high' : 3.0},
        {'unacc' : 1.0,
         'acc' : 2.0,
         'good' : 3.0,
         'vgood' : 4.0}
        ]
    
    #load BUPA set
    BUPATrainingSet = loadcsv("bupa_data_trainset.csv")
    BUPATestSet = loadcsv("bupa_data_testset.csv")
    print("BUPA set: ")
    printDetails(BUPATrainingSet, BUPA_CLASS_INDEX)
    print()
    runClassification(BUPATrainingSet, BUPATestSet, BUPA_CLASS_INDEX, MIN_K, MAX_K)
    print("\nReversing testing and training sets...")
    runClassification(BUPATestSet, BUPATrainingSet, BUPA_CLASS_INDEX, MIN_K, MAX_K)
    print()

    #load car set
    carTrainingSet = loadcsv("car_data_trainset.csv")
    carTestSet = loadcsv("car_data_testset.csv")
    print("Car set: ")
    printDetails(carTrainingSet, CAR_CLASS_INDEX)
    print()
    #convert car data to purely numerical values
    carTrainingSet = convertData(carTrainingSet, CAR_VALUES)
    carTestSet = convertData(carTestSet, CAR_VALUES)
    runClassification(carTrainingSet, carTestSet, CAR_CLASS_INDEX, MIN_K, MAX_K)
    print("\nReversing testing and training sets...")
    runClassification(carTestSet, carTrainingSet, CAR_CLASS_INDEX, MIN_K, MAX_K)

    

def runClassification(trainingSet, testSet, classIndex, minK, maxK):
    #separate out classification data
    rawTrainingSet = separateClass(trainingSet, classIndex)
    rawTestSet = separateClass(testSet, classIndex)
    #get normalized training sets and zip with classification data
    #zipped with classification data so classification order maintained when sorted
    normalizedTrainingSet = list(zip(normalize(rawTrainingSet[0]), rawTrainingSet[1]))
    rawTrainingSet = list(zip(rawTrainingSet[0], rawTrainingSet[1]))
    #get normalized test sets
    #leave testing classifiers unzipped since order will remain the same, and simplifies element iteration
    normalizedTestSet = (normalize(rawTestSet[0]), rawTestSet[1])

    sets = [(('e',  normalizedTestSet, normalizedTrainingSet), "Euclidian Distance"),
            (('c',  normalizedTestSet, normalizedTrainingSet), "Cosine Similarity"),
            (('p',  rawTestSet, rawTrainingSet), "Pearson Correlation")]

    maxAccuracy = (0, 0, 0)
    for k in range(minK, maxK + 1):
        print("k: " + str(k))
        for set in sets:
            accuracy = 0
            classification = []
            print("Function: " + set[1])
            classification = classify(k, *set[0])
            accuracy = getAccuracy(classification, set[0][1][1])
            print("Accuracy: " + str(accuracy))
            if accuracy > maxAccuracy[0]:
                maxAccuracy = (accuracy, k, set[1])
        print()
    print("Maximum Accuracy: " + str(maxAccuracy))
    return

def printDetails(dataSet, classIndex):
    numClass = numClassification(set[classIndex] for set in dataSet)
    numAttributes = len(dataSet[0])
    print("Data set contains " + str(len(numClass)) + " classifiers:")
    for classifier in numClass:
        print(str(classifier) + ": " + str(numClass[classifier]))
    print("Total attributes: " + str(numAttributes))

def numClassification(classifiers):
    found = {}
    for classifier in classifiers:
        if classifier not in found:
            found[classifier] = 1
        else:
            found[classifier] += 1
    return found

def convertData(dataSet, valueIndex):
    set = copy.deepcopy(dataSet)
    numAttributes = len(dataSet[0])
    for element in set:
        for i in range(numAttributes):
            if element[i] in valueIndex[i]:
                element[i] = valueIndex[i][element[i]]
    return set

def offsetEpsi(numericalDataSet):
    for row in numericalDataSet:
        for element in row:
            element += EPSI
    return numericalDataSet

def getAccuracy(computedClassification, knownClassification):
    accurate = 0
    for classifier in zip(computedClassification, knownClassification):
                #print(classifier[0])
                if classifier[0] == classifier[1]:
                    accurate += 1
    return accurate / len(knownClassification) * 100

def classify(k, distanceFunction, testSet, trainingSet):
    classification = []
    for point in testSet[0]:
        average = 0
        weight = 0
        for classifier in knn(distanceFunction, trainingSet, point, k):
            average += classifier
        average /= k
        classification.append(int(round(average)))
    return classification

def separateClass(dataSet, classIndex):
    set = copy.deepcopy(dataSet)
    classifier = []
    for item in set:
        classifier.append(item.pop(classIndex))
    #offset non-classifiers by epsi to avoid zero division
    return (offsetEpsi(set), classifier)

def normalize(set):
    numAttributes = len(set[0])
    numRows = len(set)
    setRange = []
    norm = [[] for _ in range(numRows)]
    for i in range(numAttributes):
        setRange.append((max(set, key = lambda row: row[i])[i], min(set, key = lambda row: row[i])[i]))
        for j in range(numRows):
            #add epsi to avoid 0 division
            norm[j].append(((set[j][i] - setRange[i][1]) / (setRange[i][0] - setRange[i][1])) + EPSI)
    return norm


#load a CSV file and return list of vectors
def loadcsv(fname):
    "loads a csv file"
    rlist = []
    test = 0
    with open(fname, 'r') as fvar:
        reader = csv.reader(fvar)
        for row in reader:
            row[:] = [(float(item)) if testNum(item) else item for item in row]
            rlist.append(row)
    return rlist

def testNum(f):
    try:
        float(f)
        return True
    except ValueError:
        return False


#Euclidean Distance Function 2 vectors
def edist2(vector1, vector2):
    "Euclidean Distance Function for 2 Vectors"
    #assumes both vectors same dimensionality
    dist = 0
    length = len(vector1)
    for i in range(length):
        dist += pow((vector1[i] - vector2[i]), 2)
    return math.sqrt(dist)

#Dot Product
def dprod(vector1, vector2):
    "Dot Product"
    if len(vector1) != len(vector2):
        return 0
    return sum(i[0] * i[1] for i in zip(vector1, vector2))

#Euclidean Distance Function 1 vector
def edist(vector1):
    "Euclidean Distance Function for 1 Vector"
    dist = 0
    for item in vector1:
        dist += pow(item, 2)
    return math.sqrt(dist)

#Cosine Similarity Function
def cosim(vector1, vector2):
    "Cosine Similarity Function"
    return dprod(vector1, vector2) / (edist(vector1) * edist(vector2))

#Pearsons Correlation
def pearson(vector1, vector2):
    "Pearson Correlation"
    xbar = 0
    ybar = 0
    sxy = 0
    sx = 0
    sy = 0
    n = len(vector1)
    for i in range(0, n):
        xbar += vector1[i]
        ybar += vector2[i]
    xbar /= n
    ybar /= n
    for i in range(0, n):
        sx += pow(vector1[i] - xbar, 2)
    sx /= n - 1
    sx = math.sqrt(sx)
    for i in range(0, n):
        sy += pow(vector2[i] - ybar, 2)
    sy /= n - 1
    sy = math.sqrt(sy)
    if sx == 0 or sy == 0:
        return 0
    for i in range(0, n):
        sxy += (vector1[i] - xbar) * (vector2[i] - ybar)
    sxy /= n - 1
    return sxy / (sx * sy)

#get neighbors
def knn(distanceFunction, dataSet, dataPoint, k):
    "Get Neighbors"
    dist = []
    if distanceFunction == 'e':
        for i in range(0, len(dataSet)):
            dist.append((dataSet[i], edist2(dataPoint, dataSet[i][0])))
        dist.sort(key = operator.itemgetter(1))
        return [element[0][1] for element in dist[:k]]
    elif distanceFunction == 'c':
        for i in range(0, len(dataSet)):
            dist.append((dataSet[i], cosim(dataPoint, dataSet[i][0])))
        dist.sort(key = operator.itemgetter(1))
        return [element[0][1] for element in dist[-k:]]
    elif distanceFunction == 'p':
        for i in range(0, len(dataSet)):
            dist.append((dataSet[i], abs(pearson(dataPoint, dataSet[i][0]))))
        dist.sort(key = operator.itemgetter(1))
        return [element[0][1] for element in dist[-k:]]
    else:
        raise ValueError("Invalid distance function type")

#main block
main()