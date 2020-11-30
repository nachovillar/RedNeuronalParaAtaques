# Importamos la data como matriz
from train_input import ye, xe
from test_input import yv, xv

from numpy import hstack, ones, append
from PSO import *
import pandas as pd



D, N = xe.shape
bias = ones((1, N))

Xe = append(xe, bias, axis=0)

pso = PSO(D+1, Xe, ye)
gBest, wBest, MSE = pso.PSO()

gBest.to_csv('data/gBest.txt', sep=',')
wBest.to_csv('data/wBest.txt', sep=',')
MSE.to_csv('data/MSE.txt', sep=',')


