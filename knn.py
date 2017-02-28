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
    #assuming the positive classifier in the BUPA set is 2.0 (not exactly sure if that is correct)
    BUPA_POS_CLASS = [2.0]
    #assuming the split between negative and positive values is between unacceptable and acceptable
    CAR_POS_CLASS = [2.0, 3.0, 4.0]

    MIN_K = 15
    MAX_K = 15

    #numerical conversions for non-numerical car data
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
    #print set details
    print("BUPA set: ")
    printDetails(BUPATrainingSet, BUPA_CLASS_INDEX)
    print()
    runClassification(BUPATrainingSet, BUPATestSet, BUPA_CLASS_INDEX, True, MIN_K, MAX_K, BUPA_POS_CLASS)
    print("\nReversing testing and training sets...\n")
    printDetails(BUPATestSet, BUPA_CLASS_INDEX)
    print()
    runClassification(BUPATestSet, BUPATrainingSet, BUPA_CLASS_INDEX, True, MIN_K, MAX_K, BUPA_POS_CLASS)
    print()

    #load car set
    carTrainingSet = loadcsv("car_data_trainset.csv")
    carTestSet = loadcsv("car_data_testset.csv")
    print("Car set: ")
    printDetails(carTrainingSet, CAR_CLASS_INDEX)
    print()
    #convert car data to purely numerical values
    convertedCarTrainingSet = convertData(carTrainingSet, CAR_VALUES)
    convertedCarTestSet = convertData(carTestSet, CAR_VALUES)
    runClassification(convertedCarTrainingSet, convertedCarTestSet, CAR_CLASS_INDEX, False, MIN_K, MAX_K, CAR_POS_CLASS)
    print("\nReversing testing and training sets...\n")
    printDetails(carTestSet, CAR_CLASS_INDEX)
    print()
    runClassification(convertedCarTestSet, convertedCarTrainingSet, CAR_CLASS_INDEX, False, MIN_K, MAX_K, CAR_POS_CLASS)

    
#run knn classification for the given training and test sets, with classifications at the given index, for the given k range, and print results
#whether the classifier is nominal or not is indicated by the flag nominal
#an optional parameter positive classifiers provides a list of classifiers considered "positive" for evaluating recall and precision, if not provided only accuracy calculated
def runClassification(trainingSet, testSet, classIndex, nominal, minK, maxK, positiveClassifiers = None):
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
    #sets to run classification on
    sets = [(('e',  normalizedTestSet, normalizedTrainingSet), "Euclidian Distance"),
            (('c',  normalizedTestSet, normalizedTrainingSet), "Cosine Similarity"),
            (('p',  rawTestSet, rawTrainingSet), "Pearson Correlation")]

    maxAccuracy = (0, 0, 0)
    maxRecall = (0, 0, 0)
    maxPrecision = (0, 0, 0)
    #classify for specified k range
    for k in range(minK, maxK + 1):
        print("k: " + str(k))
        #classify each set type
        for set in sets:
            classification = []
            print("Function: " + set[1])
            #get classifications
            classification = classify(k, nominal, *set[0])
            #calculate accuracy of classification
            if(positiveClassifiers is None):
                accuracy = getAccuracy(classification, set[0][1][1])
                print("Accuracy: " + str(accuracy))
                if accuracy > maxAccuracy[0]:
                    maxAccuracy = (accuracy, k, set[1])
            else:
                print()
                arp = ARPStats(getARP(classification, set[0][1][1], positiveClassifiers))
                print("Classifiers:\n")
                for classifier in arp:
                    print(str(classifier[0]) + ": ")
                    print("Accuracy: " + str(classifier[1]))
                    print("Recall: " + str(classifier[2]))
                    print("Precision: " + str(classifier[3]) + "\n")
                print("Confusion Matrix:")
                printConfusion(*arp[len(arp) - 1][4])
                print()
                if arp[len(arp) - 1][1] > maxAccuracy[0]:
                    maxAccuracy = (arp[len(arp) - 1][1], k, set[1])
                if arp[len(arp) - 1][2] > maxRecall[0]:
                    maxRecall = (arp[len(arp) - 1][2], k, set[1])
                if arp[len(arp) - 1][3] > maxPrecision[0]:
                    maxPrecision = (arp[len(arp) - 1][3], k, set[1])
            #determine which classification technique and k value in range gave best accuracy
            
        print()
    if(positiveClassifiers is None):
        print("Maximum Accuracy: " + str(maxAccuracy))
    else:
        print("Maximum Accuracy: " + str(maxAccuracy))
        print("Maximum Recall: " + str(maxRecall))
        print("Maximum Precision: " + str(maxPrecision))
    return

def printDetails(dataSet, classIndex):
    #get number of each class
    numClass = numClassification(set[classIndex] for set in dataSet)
    #get number of attributes
    numAttributes = len(dataSet[0])
    #print details
    print("Data set contains " + str(len(numClass)) + " classifiers:")
    for classifier in numClass:
        print(str(classifier) + ": " + str(numClass[classifier]))
    print("Total attributes: " + str(numAttributes))

def numClassification(classifiers):
    found = {}
    #count number of each classifier in given set
    for classifier in classifiers:
        if classifier not in found:
            found[classifier] = 1
        else:
            found[classifier] += 1
    return found

#convert data to numerical data based on provided value conversion dictionary
#for classification to work properly classifiers must be
def convertData(dataSet, valueIndex):
    set = copy.deepcopy(dataSet)
    numAttributes = len(dataSet[0])
    #convert data
    for element in set:
        for i in range(numAttributes):
            if element[i] in valueIndex[i]:
                element[i] = valueIndex[i][element[i]]
    return set

#offset values by global value epsilon
def offsetEpsi(numericalDataSet):
    for row in numericalDataSet:
        for element in row:
            element += EPSI
    return numericalDataSet

def printConfusion(TP, TN, FP, FN):
    print("\t\t\t\tPredicted")
    print("\t\t\tPositive\tNegative")
    print("Actual\tPositive\t" + str(TP) + "\t\t" + str(FN))
    print("\tNegative\t" + str(FP) + "\t\t" + str(TN))

def ARPStats(classifiers):
    stats = []
    totalTP = 0
    totalTN = 0
    totalFP = 0
    totalFN = 0
    for classifier in classifiers:
        TP = classifiers[classifier][0]
        TN = classifiers[classifier][1]
        FP = classifiers[classifier][2]
        FN = classifiers[classifier][3]
        accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
        try:
            recall = TP / (TP + FN) * 100
        except ZeroDivisionError:
            recall = 'N/A'
        try:
            precision = TP / (TP + FP) * 100
        except ZeroDivisionError:
            recall = 'N/A'
        #divide by 2 for each since all added twice
        totalTP += TP / 2
        totalTN += TN / 2
        totalFP += FP / 2
        totalFN += FN / 2
        stats.append((classifier, accuracy, recall, precision))
    totalAccuracy = (totalTP + totalTN) / (totalTP + totalTN + totalFP + totalFN) * 100
    totalRecall = totalTP / (totalTP + totalFN) * 100
    totalPrecision = totalTP / (totalTP + totalFP) * 100
    stats.append(('Total', totalAccuracy, totalRecall, totalPrecision, (totalTP, totalTN, totalFP, totalFN)))
    return stats

#compute accuracy, recall, and precision of a classification
def getARP(computedClassification, knownClassification, positiveClassifers):
    classifiers = {}
    #count classifications in each category for each class
    for classifier in zip(computedClassification, knownClassification):
        if classifier[1] not in classifiers:
            classifiers[classifier[1]] = [0, 0, 0, 0]
        if classifier[0] not in classifiers:
            classifiers[classifier[0]] = [0, 0, 0, 0]
        if classifier[0] == classifier[1]:
            if classifier[0] in positiveClassifers:
                classifiers[classifier[0]][0] += 1
                classifiers[classifier[1]][0] += 1
            else:
                classifiers[classifier[0]][1] += 1
                classifiers[classifier[1]][1] += 1
        else:
            if classifier[0] in positiveClassifers:
                classifiers[classifier[0]][2] += 1
                classifiers[classifier[1]][2] += 1
            else:
                classifiers[classifier[0]][3] += 1
                classifiers[classifier[1]][3] += 1
    return classifiers


#compute accuracy of a classification
def getAccuracy(computedClassification, knownClassification):
    accurate = 0
    #count accurate classifications
    for classifier in zip(computedClassification, knownClassification):
        #print(classifier[0])
        if classifier[0] == classifier[1]:
            accurate += 1
    #calculate percent accuaracy
    return float(accurate) / len(knownClassification) * 100

#classify data with k nearest neighbors function
def classify(k, nominal, distanceFunction, testSet, trainingSet):
    classification = []

    #classify on greatest number of similar classifiers if classifier nominal
    if nominal:
        classifiers = {}
        #enumerate possible classifiers
        for point in trainingSet:
            if point[1] not in classifiers:
                classifiers[point[1]] = 0
        #classify each point in set
        for point in testSet[0]:
            average = 0
            for classifier in knn(distanceFunction, trainingSet, point, k):
                classifiers[classifier] += 1
            classification.append(max(classifiers, key = lambda value: classifiers[value]))
            #reset counts
            for classifier in classifiers:
                classifiers[classifier] = 0

    #classifiy on average of classifiers if classifier ordinal
    else:
        classifiers = []
        #enumerate possible classifiers
        for point in trainingSet:
            if point[1] not in classifiers:
                classifiers.append(point[1])
        for point in testSet[0]:
            average = 0
            #enumerate possible classifiers
            for classifier in knn(distanceFunction, trainingSet, point, k):
                average += classifier
            #ensure average properly computed by casting
            average = float(average) / k
            #find the classifier that is closest to the average
            closestClassifier = 0
            minDifference = math.inf
            classification.append(min(classifiers, key = lambda val: abs(average - val)))
            #for classifier in classifiers:
            #    if abs(average - classifier) < minDifference:
            #        closestClassifier = classifier
            #        minDifference = abs(average - classifier)
            #classification.append(closestClassifier)

    #return list of classifications for each point
    return classification

#seperate out classification attribute from data set
def separateClass(dataSet, classIndex):
    set = copy.deepcopy(dataSet)
    classifier = []
    for item in set:
        classifier.append(item.pop(classIndex))
    #offset non-classifiers by epsi to avoid zero division
    return (offsetEpsi(set), classifier)

#normalize a data set by placing values between epsilon and 1 + epsilon
def normalize(set):
    numAttributes = len(set[0])
    numRows = len(set)
    setRange = []
    norm = [[] for _ in range(numRows)]
    #find maxes and mins of each attribute
    for i in range(numAttributes):
        setRange.append((max(set, key = lambda row: row[i])[i], min(set, key = lambda row: row[i])[i]))
        for j in range(numRows):
            #normalize to set range, with the minimum value at epsilon, and the maximum at 1 + epsilon
            #range offset by epsilon to avoid 0 division if a 0 vector is created
            norm[j].append(((set[j][i] - setRange[i][1]) / (setRange[i][0] - setRange[i][1])) + EPSI)
    return norm


#load a CSV file and return list of vectors
def loadcsv(fname):
    "loads a csv file"
    rlist = []
    test = 0
    #open file for reading
    with open(fname, 'r') as fvar:
        reader = csv.reader(fvar)
        #read each row into array and attempt to convert attributes to numbers if numerical values
        for row in reader:
            row[:] = [(float(item)) if testNum(item) else item for item in row]
            rlist.append(row)
    return rlist

#test if given value is a number
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

#get k nearest neighbors
def knn(distanceFunction, dataSet, dataPoint, k):
    "Get Neighbors"
    dist = []
    #run using specified distance function
    if distanceFunction == 'e':
        #evaluate distance to points in training set
        for i in range(0, len(dataSet)):
            dist.append((dataSet[i], edist2(dataPoint, dataSet[i][0])))
        #sort points and select nearest points based on type of distance function
        dist.sort(key = operator.itemgetter(1))
        return [element[0][1] for element in dist[:k]]
    elif distanceFunction == 'c':
        for i in range(0, len(dataSet)):
            dist.append((dataSet[i], cosim(dataPoint, dataSet[i][0])))
        dist.sort(key = operator.itemgetter(1))
        return [element[0][1] for element in dist[-k:]]
    elif distanceFunction == 'p':
        for i in range(0, len(dataSet)):
            #use absolute value of distance evaluation since 0 is lowest correlation and range is on [-1, 1]
            dist.append((dataSet[i], abs(pearson(dataPoint, dataSet[i][0]))))
        dist.sort(key = operator.itemgetter(1))
        return [element[0][1] for element in dist[-k:]]
    else:
        raise ValueError("Invalid distance function type")

#main block
main()
