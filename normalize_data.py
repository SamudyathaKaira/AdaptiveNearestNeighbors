__author__ = 'sckaira'

# The data was split into
import numpy as np
import csv

training_data = np.genfromtxt('C:\\Users\\sckaira\\Desktop\Courses\\1 credit Project\\Project Implementation\\data\\pima-indians-diabetes.csv',delimiter=',')

mean = []
standardDeviation = []

# calculate mean for each feature and store it in an array called mean.
for i in range(0,9):
    total = 0
    n=0
    for row in training_data:
        total += row[i]
        n += 1
    mean.append(total/n)

# calculate standard deviation for each feature and store it in an array called standardDeviation.
for i in range(0,9):
    feature = []
    for row in training_data:
        feature.append(row[i])
    standardDeviation.append(np.std(feature))


training_matrix = [[0 for k in range(9)] for k in range(n)]

for i in range(0,9):
    j=0
    for row in training_data:
        training_matrix[j][i] = (row[i]-mean[i])/standardDeviation[i]
        j+=1

np.savetxt("C:\\Users\\sckaira\\Desktop\\Courses\\1 credit Project\\Project Implementation\\data\\norm_pima.csv", training_matrix, delimiter=",")


