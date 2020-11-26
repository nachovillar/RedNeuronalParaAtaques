# librerÃ­a para trabajar con matrices
from numpy import matrix
# LibrerÃ­a para acceder a la inversa de las matrices
from scipy import linalg
# LibrerÃ­a para el manejo de archivos .csv
import pandas as pd

#data test
dataFrame = pd.read_csv('data/KDDTest', sep = ',')
#data entranada 
dataTrain = pd.read_csv('data/pesos',sep = ',')


dataMatrix = matrix(dataFrame)
dataPesos = matrix (dataTrain)