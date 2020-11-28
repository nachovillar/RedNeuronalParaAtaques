# librería para trabajar con matrices
from numpy import matrix
import numpy as np
from numpy import shape, matlib
# Librería para acceder a la inversa de las matrices
from scipy import linalg
# Librería para el manejo de archivos .csv
import pandas as pd

dataFrame = pd.read_csv('data/KDDTrain+_20Percent.txt', sep = ',')

dataMatrix = matrix(dataFrame)

# print(dataMatrix)

print(np.exp(1))
# libreria para trabajar con matrices
from numpy import matrix, random
# LibrerÃia para acceder a la inversa de las matrices
from scipy import linalg
# Libreri­a para el manejo de archivos .csv
import pandas as pd

#data entrenamiento
dataFrameTrain = pd.read_csv('data/KDDTrain+_20Percent.txt', header=None, sep = ',')

yeDataFrame = dataFrameTrain[41]

xeDataFrame = dataFrameTrain
xeDataFrame.pop(41)
xeDataFrame.pop(42)

for i in range(len(yeDataFrame)):
  if yeDataFrame[i] == 'normal':
    yeDataFrame[i] = 1
  else:
    yeDataFrame[i] = -1

ye = matrix(yeDataFrame)
xe = matrix(xeDataFrame)




###########################################




