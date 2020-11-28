# libreria para trabajar con matrices
from numpy import matrix, nditer
# LibrerÃia para acceder a la inversa de las matrices
from scipy import linalg
# Libreri­a para el manejo de archivos .csv
import pandas as pd

#data test
dataFrameTrain = pd.read_csv('data/KDDTrain+_20Percent.txt', header=None, sep = ',')
#data entrenamiento
dataFrameTest = pd.read_csv('data/KDDTest+.txt', header=None, sep = ',')


yeDataFrame = dataFrameTrain[41]

xeDataFrame = dataFrameTest
xeDataFrame.pop(41)

for i in range(len(yeDataFrame)):
  if yeDataFrame[i] == 'normal':
    yeDataFrame[i] = 1
  else:
    yeDataFrame[i] = -1

ye = matrix(yeDataFrame)
xe = matrix(xeDataFrame)

print(xe)


