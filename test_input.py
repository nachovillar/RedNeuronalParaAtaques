# librería para trabajar con matrices
from numpy import matrix, shape
# Librería para el manejo de archivos .csv
import pandas as pd

#data entrenamiento
dataFrameTrain = pd.read_csv('data/kddtest.txt', header=None, sep = ',')

yeDataFrame = dataFrameTrain[41]

xeDataFrame = dataFrameTrain
xeDataFrame.pop(41)
xeDataFrame.pop(42)

yv = matrix(yeDataFrame)
xv = matrix(xeDataFrame)