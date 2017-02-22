"K-Nearest-Neighbors Classifier with test data and misc. other algorithms for comparison"
import csv
import math
import statistics
#functions
#main function
def main():
    "main function"
    trainingset = loadcsv("car_data_trainset.csv")
    testset = loadcsv("car_data_testset.csv")
    print(testset, trainingset)
    return
#load a CSV file and return list of vectors
def loadcsv(fname):
    "loads a csv file"
    rlist = []
    with open(fname, 'r') as fvar:
        reader = csv.reader(fvar)
        for row in reader:
            rlist.insert(0, row)
    return rlist
#Euclidean Distance Function 2 vectors
def edist2(vector1, vector2):
    "Euclidean Distance Function for 2 Vectors"
    #assumes both vectors same dimensionality
    dist = 0
    length = len(vector1)
    for i in range(length):
        dist += pow((vector1[i] - vector2[i]), length)
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
#Covariance Function
def cov(vector1, vector2):
    "Covariance Function"
    return
#main block
main()
