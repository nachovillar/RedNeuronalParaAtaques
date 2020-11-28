import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from numpy import matrix


trainData = pd.read_csv('data/KDDTrain+_20Percent.txt', sep=',', header = None)
testData = pd.read_csv('data/KDDTest+.txt', sep=',', header = None)

a = 0.1
b = 0.99

encoder = preprocessing.LabelEncoder()


for col in range(1, 4):
    trainData[col] = encoder.fit_transform(trainData[col])
    testData[col] = encoder.fit_transform(testData[col])


trainData[41] = trainData[41] == 'normal'
trainData[41] = trainData[41].replace(True, 1)
trainData[41] = trainData[41].replace(0, -1)

testData[41] = testData[41] == 'normal'
testData[41] = testData[41].replace(True, 1)
testData[41] = testData[41].replace(False, -1)


xeTrain = trainData.loc[:, :40]
xeTest = testData.loc[:, :40]

print(xeTrain)
norm_xeTrain = (b - a) * ((xeTrain - xeTrain.min()) / (xeTrain.max() - xeTrain.min())) + a
norm_xeTest = (b - a) * ((xeTest - xeTest.min()) / (xeTest.max() - xeTest.min())) + a

print(norm_xeTrain)


kddtrain = norm_xeTrain.join(trainData.loc[:, 41:])
kddtest = norm_xeTest.join(testData.loc[:, 41:])

print(kddtrain)
kddtrain.to_csv('data/kddtrain.txt', sep=',', header = None, index = False)
kddtest.to_csv('data/kddtest.txt', sep=',', header = None, index = False)
