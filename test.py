from numpy import matrix, shape, matlib
import numpy as np
# Librería para el manejo de archivos .csv
import pandas as pd
from test_input import yv, xv
from numpy import hstack, ones, append
from PSO import *


#carga los datas de test, y se importan del test_input



D, N = xv.shape
bias = ones((1, N))

Xe = append(xv, bias, axis=0)

pso = PSO(D+1, Xe, yv)
gBest, wBest, MSE = pso.PSO()



#cargan los datos ya entrenados
DataFrameW1= pd.read('data/gBest.txt', sep=',')
DataFrameW2 = pd.read('data/wBest.txt', sep=',')
DtaFrameMSE = pd.read('data/MSE.txt', sep=',')


accuracy, Fscore = metrica(z, ye)

print (accuracy)
print (Fscore)

fscore = pd.DataFrame (Fscore,accuracy)
fscore.to_csv('data/fscore.txt', sep=',')








#@yh = yEsperado
#@yo = yObtenido

def metrica(yh, yo):
    TP = 0
    FN = 0
    FP = 0
    for i in range(len(yh)):
        if yh[i] and yo[i] == 1:
            TP += 1 #verdadero positivo
        if yh[i] == -1 and yo[i] == 1:
            FP += 1 #falso positivo
        if yh[i] == 1 and yo[i] == -1:
            FN += 1 #falso negativo
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f_score = 2/ ((1/recall)+(1/precision))
    
    return precision, f_score

def activation (self, x, w):
    z = np.dot(w, x)
    h = ((2/(1+ exp(-z)))-1)

    return h