# librería para trabajar con matrices
from numpy import matrix
import numpy
# Librería para acceder a la inversa de las matrices
from scipy import linalg
# Librería para el manejo de archivos .csv
import pandas as pd

dataFrame = pd.read_csv('data/KDDTrain+_20Percent.txt', sep = ',')

dataMatrix = matrix(dataFrame)

# print(dataMatrix)

print(numpy.exp(1))
