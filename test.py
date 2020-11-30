from numpy import matrix, shape, matlib
import numpy as np
# Librería para el manejo de archivos .csv
import pandas as pd
from test_input import yv, xv

D, N = xv.shape
bias = ones((1, N))

Xe = append(xv, bias, axis=0)

pso = PSO(D+1, Xe, yv)
gBest, wBest, MSE = pso.PSO()

z = forward(xv, w1, bias, w2)
accuracy, Fscore = metrica(z, ye)

print (accuracy)
print (Fscore)
#falta cargar los pesos 







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