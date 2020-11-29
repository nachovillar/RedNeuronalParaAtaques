# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:13:28 2020

@author: coott
"""
#Importamos la data como matriz y como frame
from train_input import ye, xe
# from param_config import Xe, Ye, L, C
from numpy import random, shape, sqrt, matlib, transpose, exp, eye
# Librería para acceder a la inversa de las matrices
from scipy import linalg
from param_config import *
import numpy as np
import math
from sklearn.metrics import mean_squared_error as mse



def __init__(self, D, xe, ye, C):
    self.maxIter = MAX_ITERACIONES
    self.np = PARTICULAS
    self.nh = NODOS_OCULTOS
    self.X = None
    self.D = D
    self.xe = xe  
    self.ye = ye
    self.C = C
    self.w1 = None

    #Inicializacion de la poblacion
    self.ini_swarm(PARTICULAS,NODOS_OCULTOS,D)

def _ini_config_swarm(self,PARTICULAS, NODOS_OCULTOS,):
        iter = 0
        alfa = np.zeros(self.maxIter)
        amax=0.95
        amin=0.1
        for p in range(self.maxIter):
            alfa[p] = amax- ((amax - amin)/self.maxIter)*p
            
        pBest = np.zeros((self.np, self.D*self.nh))
        pFitness = np.ones(self.np)*100000
        gBest = np.ones(self.D*self.nh)
        wBest = np.zeros(self.nh)
        gFitness = 1000000000
        MSE = np.zeros((self.maxIter))

#PRoceso de aprendizaje
def PSO(self):
    iter=0
    _ini_config_swarm()
    for iter in range(self.maxIter):
        new_pFitness, newBeta = self.fitness()
        pBest, pFitness, gBest, gFitness, wBest = self.upd_particle(self.X, pBest, pFitness, gBest,gFitness,new_pFitness, newBeta, wBest)

    MSE[iter] = gFitness
    i=1
    j=1
    for i in Np:
        for j in Dim:
            z1[1,j] = c1 *rand *(pBest[i,j]-x[i,j])# local
            z2[1,j] = c2 *rand *(pBest[i,j]-x[i,j])# global
               
    V=alpha(iter)*V+z1+z2
    x=x+V
    return (gBest ,wBest,MSE)
 
# funcion de costo 
def fitness(self):  
    w2 = np.zeros((self.np, self.nh), dtype=float)
    MSE = np.zeros(self.np , dtype=float)
    for i in range(self.np):# se evalua la red por cada red neuronal
        p = self.X[i]# se tomo una particula con todos sus datos(vector)
        w1 = np.reshape(p, (self.nh, self.D)) #trasnformo a una matriz de los pesos ocultos
        H = self.activation(self.xe, w1)# @h = matris =nodo_ocultodo x muestra de entrenamiento
        w2[i] = self.mlp_pinv(H)
        ze = np.matmul(w2[i],H)
        MSE[i] = math.sqrt(mse(self.ye, ze))
    return MSE, w2
#Pesos de salida, usando seudo inversa
def mlp_pinv(self, H):
    L,N = H.shape
    yh = np.matmul(np.transpose(self.ye),np.transpose(H))
    hh = np.matmul(H,np.transpose(H))
    hh = hh + (np.eye(hh.shape[0])/self.C)
    w2 = np.matmul(np.transpose(yh),np.linalg.pinv(hh))

    return w2


def upd_particle(self, X, pBest,pFitness, gBest, gFitness, New_pFitness, newBeta, wBest):
    for i in range(self.np):
        if (New_pFitness[i] < pFitness[i]):
            pFitness[i] = New_pFitness[i]
            pBest[i][:] = X[i, :]
    New_gFitness = min(pFitness)
    idx = np.argmin(pFitness)
    if (New_gFitness < gFitness):
        gFitness = New_gFitness
        gBest = pBest[idx][:]
        wBest = newBeta[idx][:]

    return pBest, pFitness, gBest, gFitness, wBest
#aun no entiendo que valor tiene m y como se mueve j
#def activation (x,w):
#    z=0
#    for i =1 to m:
#        z= z+ x[n,i]*w[j,i]
#    h=((2/(1+ exp(-z)))-1)
#    return h



    