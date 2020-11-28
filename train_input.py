# librería para trabajar con matrices
from numpy import matrix, shape
# Librería para el manejo de archivos .csv
import pandas as pd

#data entrenamiento
dataFrameTrain = pd.read_csv('data/kddtrain.txt', header=None, sep = ',')

yeDataFrame = dataFrameTrain[41]

xeDataFrame = dataFrameTrain
xeDataFrame.pop(41)
xeDataFrame.pop(42)

ye = matrix(yeDataFrame)
xe = matrix(xeDataFrame)





###########################################




