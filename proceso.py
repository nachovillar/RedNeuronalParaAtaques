import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


trainData = pd.read_csv('data/KDDTrain+_20Percent.txt', sep=',')
testData = pd.read_csv('data/KDDTest+.txt', sep=',')

a = 0.1
b = 0.99

encoder = preprocessing.LabelEncoder()

for col in range(1, 3):
    trainData[col] = encoder.fit_transform(trainData[col])
    testData[col] = encoder.fit_transform(testData[col])

if trainData[41] == 'normal':
    trainData[41] = 1
else:
    trainData[41] = -1

if testData[41] == 'normal':
    testData[41] = 1
else:
    trainData[41] = -1

xeTrain = trainData.loc[:, 40]
xeTest = testData.loc[:, 40]

norm_xeTrain = (b - a) * ((xeTrain - xeTrain.min()) / (xeTrain.max() - xeTrain.min())) + a
norm_xeTest = (b - a) * ((xeTest - xeTest.min()) / (xeTest.max() - xeTest.min())) + a

kddtrain = norm_xeTrain.join(trainData.loc[41, 42])
kddtest = norm_xeTest.join(testData.loc[41, 42])

kddtrain.to_csv('kddtrain.txt', sep=',')
kddtest.to_csv('kddtest.txt', sep=',')
