__author__ = 'sckaira'

import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import math
import random

knnResult = []
annResult = []
annFResult = []
annNResult = []

def knn( dataFile, targetClassRow, metric, p, class1, class2):
    global knnResult
    global annResult
    global errorResultKnn
    global errorResultAnn

    knnResult = []
    annResult = []
    errorResultKnn=[]
    errorResultAnn=[]

    training_data = np.genfromtxt('C:\\Users\\sckaira\\Desktop\\Courses\\1 credit Project\\Project Implementation\\data\\' + dataFile +".csv",delimiter=',')

    np.random.shuffle(training_data)
    # define a function and generate a graph for each of the knn with euclidean distances

    yTraining =[] # class labels of the training
    for row in training_data:
        yTraining.append(row[targetClassRow])
    X = np.delete(training_data, targetClassRow, axis=1)

    dv10 = training_data.__len__()/10
    # 10 fold Cross Validation

    for k in range(1,51):
        accuracy = []
        for i in range(0,10):
            #Training all data expect Si called SiRest
            sIRest = []
            sIRestY = []
            sI = []
            sIY = []
            startIndex = i * dv10
            endIndex = (i+1) * dv10
            sIRest= np.delete(X, np.s_[startIndex:endIndex], axis=0)
            sIRestY = np.delete(yTraining, np.s_[startIndex:endIndex], axis=0)
            neigh = KNeighborsClassifier(n_neighbors=k, p=p, metric=metric)
            #neigh = KNeighborsClassifier(n_neighbors=k, p=1, metric='manhattan')
            neigh.fit(sIRest, sIRestY)

            #Test data == Si - Vaidation set
            temp = np.delete(X, np.s_[0:startIndex], axis=0)
            sI= np.delete(temp, np.s_[dv10:X.__len__()], axis=0)
            # actual class of the Validation data
            sIY = np.delete(np.delete(yTraining, np.s_[0:startIndex], axis=0), np.s_[dv10:X.__len__()], axis=0)

            accuracy.append(sklearn.metrics.accuracy_score(sIY, neigh.predict(sI)))
            #accuracy.append(neigh.score(sI,sIY)) # same result as predict
        error= np.mean(accuracy)
        error = (1.00 - error)*100
        knnResult.append(k)
        errorResultKnn.append(error)
        print "k", k, "err=",error
        error = 0.0
    if(p == 2):
        AnnEuclidean(dataFile,targetClassRow,metric,p,class1, class2)
    else:
        AnnManhattan(dataFile,targetClassRow,metric,p,class1, class2)

    print "Errors of Knn", errorResultKnn
    print "Errors of Ann", errorResultAnn
    plt.plot(knnResult, errorResultKnn)
    plt.plot(annResult, errorResultAnn)
    plt.title('Knn-Ann ' + dataFile + " - " + metric)
    plt.savefig('knn-ann-' + dataFile + "-" + metric + '.pdf')
    plt.grid(True)


def AnnManhattan( dataFile, targetClassRow, metric, p, class1, class2):
    global knnResult
    global annResult
    global errorResultKnn
    global errorResultAnn

    training_data = np.genfromtxt('C:\\Users\\sckaira\\Desktop\\Courses\\1 credit Project\\Project Implementation\\data\\' + dataFile+".csv",delimiter=',')
    np.random.shuffle(training_data)
    yTraining =[] # class labels of the training
    for row in training_data:
        yTraining.append(row[targetClassRow])

    X = np.delete(training_data, targetClassRow, axis=1)

    dv10 = training_data.__len__()/10
    # 10 fold Cross Validation
    predictedClass = []

    for k in range(1,51):
        accuracy = []
        for i in range(0,10):
            #Training all data expect Si called SiRest
            sIRest = []
            sIRestY = []
            sI = []
            sIY = []
            startIndex = i * dv10
            endIndex = (i+1) * dv10
            sIRest= np.delete(X, np.s_[startIndex:endIndex], axis=0)
            sIRestY = np.delete(yTraining, np.s_[startIndex:endIndex], axis=0)

            trainingclass1 =[]
            trainingclass2 =[]
            t=0
            for tclass in sIRest:
                if(sIRestY[t] == class1):
                    trainingclass1.append(tclass)
                if(sIRestY[t] == class2):
                    trainingclass2.append(tclass)
                t+=1

            # radius calculation for every element in the test set

            radiusTraining = []
            t=0
            for trainrow in sIRest:
                minDRadius =[]
                if sIRestY[t] == class1:
                    for rowclass2 in trainingclass2:# iterate each training data
                        sigdist=0
                        for rc in range(0,len(rowclass2)):
                            sigdist += abs(rowclass2[rc] - trainrow[rc])
                        minDRadius.append(sigdist)
                else:
                    for rowclass1 in trainingclass1:# iterate each training data
                        sigdist=0
                        for rc in range(0,len(rowclass1)):
                            sigdist += abs(rowclass1[rc] - trainrow[rc])
                        minDRadius.append(sigdist)

                sortRadius = [h[0] for h in sorted(enumerate(minDRadius), key=lambda x:x[1])]
                t+=1
                #epsilon = random.uniform(0,minDRadius[sortRadius[0]])
                epsilon = minDRadius[sortRadius[0]] / 10000000.0
                radius = minDRadius[sortRadius[0]] - epsilon
                radiusTraining.append(radius)

            #Test data == Si - Vaidation set
            sI= np.delete(np.delete(X, np.s_[0:startIndex], axis=0), np.s_[dv10:X.__len__()], axis=0)
            # actual class of the Validation data
            sIY = np.delete(np.delete(yTraining, np.s_[0:startIndex], axis=0), np.s_[dv10:X.__len__()], axis=0)

            for sIrow in sI:# for each test sample in sI
                t=0
                dVec =[]
                for row in sIRest:# iterate each training data
                    sigdist=0
                    for m in range(0,len(row)):
                        sigdist += abs(row[m] - sIrow[m])
                    if(radiusTraining[t] !=0):
                        sigdist = sigdist / (1.0 * radiusTraining[t])
                    dVec.append(sigdist)
                    t+=1

                sortedIndex = [h[0] for h in sorted(enumerate(dVec), key=lambda x:x[1])]

                predictClass1 =0
                predictClass2 =0

                for o in range(0,k):
                    if sIRestY[sortedIndex[o]] == class1:
                        predictClass1 += 1
                    if sIRestY[sortedIndex[o]] == class2:
                        predictClass2 += 1

                if predictClass1>predictClass2:
                    predictedClass.append(class1)
                else:
                    predictedClass.append(class2)

            accuracyS = 0
            for z in range(0,len(sIY)):
                if sIY[z] == predictedClass[z]:
                    accuracyS += 1

            predictedClass =[]
            accuracyS = accuracyS / (1.0 * len(sIY))
            accuracy.append(accuracyS)

        error = np.average(accuracy)
        error = (1.00 - error)*100
        annResult.append(k)
        errorResultAnn.append(error)
        print "k", k, "error=",error
        error = 0.0

def AnnEuclidean( dataFile, targetClassRow, metric, p, class1, class2):
    global knnResult
    global annResult
    global errorResultKnn
    global errorResultAnn
    training_data = np.genfromtxt('C:\\Users\\sckaira\\Desktop\\Courses\\1 credit Project\\Project Implementation\\data\\' + dataFile+".csv",delimiter=',')
    np.random.shuffle(training_data)
    yTraining =[] # class labels of the training
    for row in training_data:
        yTraining.append(row[targetClassRow])

    X = np.delete(training_data, targetClassRow, axis=1)

    dv10 = training_data.__len__()/10
    # 10 fold Cross Validation
    kResult = []
    errorResult =[]
    predictedClass = []

    for k in range(1,51):
        accuracy = []
        for i in range(0,10):
            #Training all data expect Si called SiRest
            sIRest = []
            sIRestY = []
            sI = []
            sIY = []
            startIndex = i * dv10
            endIndex = (i+1) * dv10
            sIRest= np.delete(X, np.s_[startIndex:endIndex], axis=0)
            sIRestY = np.delete(yTraining, np.s_[startIndex:endIndex], axis=0)

            trainingclass1 =[]
            trainingclass2 =[]
            t=0
            for tclass in sIRest:
                if(sIRestY[t] == class1):
                    trainingclass1.append(tclass)
                if(sIRestY[t] == class2):
                    trainingclass2.append(tclass)
                t+=1

            # radius calculation for every element in the test set
            #epsilon = 0.00001
            radiusTraining = []
            t=0
            for trainrow in sIRest:
                minDRadius =[]
                if sIRestY[t] == class1:
                    for rowclass2 in trainingclass2:# iterate each training data
                        sigdist=0
                        for rc in range(0,len(rowclass2)):
                            sigdist += (rowclass2[rc] - trainrow[rc]) ** 2
                        sigdist = math.sqrt(sigdist)
                        minDRadius.append(sigdist)
                else:
                    for rowclass1 in trainingclass1:# iterate each training data
                        sigdist=0
                        for rc in range(0,len(rowclass1)):
                            sigdist += (rowclass1[rc] - trainrow[rc]) ** 2
                        sigdist = math.sqrt(sigdist)
                        minDRadius.append(sigdist)

                sortRadius = [h[0] for h in sorted(enumerate(minDRadius), key=lambda x:x[1])]
                epsilon = minDRadius[sortRadius[0]] / 100000.0
                #print minDRadius[sortRadius[0]]
                #epsilon = random.uniform(0,0.000000000001)
                radius = minDRadius[sortRadius[0]] - epsilon
                radiusTraining.append(radius)
                t+=1
            #---------------------------------------------- end calc radius
            #Test data == Si - Vaidation set
            sI= np.delete(np.delete(X, np.s_[0:startIndex], axis=0), np.s_[dv10:X.__len__()], axis=0)
            # actual class of the Validation data
            sIY = np.delete(np.delete(yTraining, np.s_[0:startIndex], axis=0), np.s_[dv10:X.__len__()], axis=0)

            for sIrow in sI:# for each test sample in sI
                t=0
                dVec =[]
                for row in sIRest:# iterate each training data
                    sigdist=0
                    for m in range(0,len(row)):
                        sigdist += (row[m] - sIrow[m]) ** 2
                    if(radiusTraining[t] !=0):
                        sigdist = math.sqrt(sigdist) / (1.0 * radiusTraining[t])
                    dVec.append(sigdist)
                    t+=1

                sortedIndex = [h[0] for h in sorted(enumerate(dVec), key=lambda x:x[1])]

                predictClass1 =0
                predictClass2 =0

                for o in range(0,k):
                    if sIRestY[sortedIndex[o]] == class1:
                        predictClass1 += 1
                    if sIRestY[sortedIndex[o]] == class2:
                        predictClass2 += 1

                if predictClass1>predictClass2:
                    predictedClass.append(class1)
                else:
                    predictedClass.append(class2)

            accuracyS = 0
            for z in range(0,len(sIY)):
                if sIY[z] == predictedClass[z]:
                    accuracyS += 1

            predictedClass =[]
            accuracyS = accuracyS / (1.0 * len(sIY))
            accuracy.append(accuracyS)

        error = np.average(accuracy)
        error = (1.00 - error)*100
        annResult.append(k)
        errorResultAnn.append(error)
        print "k", k, "error=",error
        error = 0.0


#Manhattan Distance measure for all 5 datasets

#knn("breast-cancer-wisconsin", 9, 'manhattan', 1,2,4)
#knn("ionosphere", 33, 'manhattan', 1, 0, 1)
#knn("pima-indians-diabetes", 8, 'manhattan', 1, 0, 1)
#knn("bupa", 6, 'manhattan', 1, 1, 2)
#knn("sonar", 60, 'manhattan', 1, 77, 82)
#knn("norm_spam_train", 57, 'manhattan', 1, -1, 1)
#knn("heart_Statlog", 13, 'manhattan', 1, 1, 2)
#knn("heart_Statlog_norm", 13, 'manhattan', 1, 1, 2)
#knn("spam", 57, 'manhattan', 1, 0, 1)

#knn("breast-cancer-wisconsin", 9, 'euclidean', 2,2,4)
#knn("ionosphere", 33, 'euclidean', 2, 0, 1)
#knn("pima-indians-diabetes", 8, 'euclidean', 2, 0, 1)
#knn("bupa", 6, 'euclidean', 2, 1, 2)
#knn("sonar", 60, 'euclidean', 2,77 ,82 )
#knn("norm_spam_train", 57, 'euclidean', 2, -1, 1)
#knn("spam", 57, 'euclidean', 2, 0, 1)
#knn("heart_Statlog", 13, 'euclidean', 2, 1, 2)
knn("heart_Statlog_norm", 13, 'euclidean', 2, 1, 2)
