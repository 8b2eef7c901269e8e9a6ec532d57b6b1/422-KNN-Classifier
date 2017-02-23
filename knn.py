# CS422 - Data Analytics
# Jared McLean & Jared Schreiber
# 2/22/2017
"K-Nearest-Neighbors Classifier with test data"

import csv
import math
import operator

#functions

#main function
def main():
    "main function"
    trainingset = loadcsv("bupa_data_trainset.csv")
    testset = loadcsv("bupa_data_testset.csv")
    normalizedTrainingSet = normalize(trainingset)
    #print(normalizedTrainingSet)

    return

def normalize(set):
    numAttributes = len(set[0])
    numRows = len(set)
    setRange = []
    norm = [[] for _ in range(numRows)]
    for i in range(numAttributes):
        if testNum(set[0][i]):
            setRange.append((max(set, key = lambda row: row[i])[i], min(set, key = lambda row: row[i])[i]))
            for j in range(numRows):
                norm[j].append((set[j][i] - setRange[i][1]) / (setRange[i][0] - setRange[i][1]))
    return norm


#load a CSV file and return list of vectors
def loadcsv(fname):
    "loads a csv file"
    rlist = []
    test = 0
    with open(fname, 'r') as fvar:
        reader = csv.reader(fvar)
        for row in reader:
            row[:] = [float(item) for item in row if testNum(item)]
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
    return dprod(vector1, vector2) / edist(vector1) * edist(vector2)

#Pearsons Correlation
def pearsons(vector1, vector2):
    "Pearsons Correlation"
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
        sxy += (vector1[i] - xbar) * (vector2[i] - ybar)

    sxy /= n - 1

    for i in range(0, n):
        sx += pow(vector1[i] - xbar, 2)

    sx /= n - 1
    sx = math.sqrt(sx)

    for i in range(0, n):
        sy += pow(vector2[i] - ybar, 2)

    sy /= n - 1
    sy = math.sqrt(sy)

    return sxy / (sx * sy)

#get neighbors. simfunc = which similarity function
def getneighbors(matrix1, vector2, k, simfunc):
    "Get Neighbors"
    result1 = []
    result2 = []
    for i in range(0, len(matrix1)):
        result1.append((matrix1[i], simfunc(vector2, matrix1[i])))
    #at this point result = list of neighbors, not sorted
    result1.sort(key=operator.itemgetter(1))
    for index in range(0, k):
        result2.append(result1[index][0])
    return result2

#main block
#main()
#some testing stuff ::
#trainSet = [[2, 2, 2], [4, 4, 4]]
#testInstance = [5, 5, 5]
#neighbors = getneighbors(trainSet, testInstance, 1, edist2)
#print(neighbors)
