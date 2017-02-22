"K-Nearest-Neighbors Classifier with test data and misc. other algorithms for comparison"
import csv
import math
#functions
#main function
def main():
    "main function"
    loadcsv("car_data_trainset.csv")
    return
#load CSV file and return list of vectors
def loadcsv(fname):
    "loads a csv file"
    rlist = []
    with open(fname, 'rb') as fvar:
        reader = csv.reader(fvar)
        for row in reader:
            rlist.insert(0, row)
    return rlist
#main block
main()
