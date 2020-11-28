#Importamos la data como matriz y como frame
from train_input import ye, xe
from functions import activation
# from param_config import Xe, Ye, L, C
from numpy import random, shape, sqrt, matlib, transpose, exp, eye
# Librería para acceder a la inversa de las matrices
from scipy import linalg
from param_config import *


#Función para inicializar los pesos
def rand_W(next_nodes, current_nodes):
  r = sqrt(6/(next_nodes + current_nodes))
  w = random.rand(next_nodes, current_nodes)
  w = (w * 2 * r) - r
  return w
#params: Data de entrada, Data salida, Nodos ocultos, Penalidad pseudo inversa
def upd_pesos(Xe, Ye, L, C):
    [nodos_entrada, caracteristicas] = Xe.shape # numero de nodos de entrada, atributos del nodo de entrada
    w1 = rand_W(L, nodos_entrada) #Matriz de pesos ocultos
    bias = rand_W(L, 1) #Sesgo
    biasMatrix = matlib.repmat(bias, 1, caracteristicas) #Matriz para sumarle el sesgo a todos los pesos

    z = w1*Xe + biasMatrix
    H = activation(z)
    Htrnaspose = transpose(H)
    #Se calculan los pesos de salida
    yh = Ye*Htrnaspose
    hh = (H*Htrnaspose + eye(L)/C)

    inverse = linalg.inv(hh)
    w2 = yh*inverse #Pesos de Salida
    #Retornamos los pesos ocultos, sesgo y pesos de salida
    return(w1, bias, w2) 

    


w1, bias, w2 = upd_pesos(xe, ye, NODOS_OCULTOS, PENALIDAD_P_INVERSA)
w1.to_csv('matriz_pesos_ocultos.csv', sep=',')
w2.to_csv('matriz_pesos_salida.csv', sep = ',')
bias.to_csv('sesgo.csv')

# def PSO():
#     for (iter = 1 to maxIter):
#         new_pFitness, new_beta = fitness(Xe,Ye,Nh,X,Cpinv)
#         pBest,pFitness,gBest,gFintness,wBest = upd_particle(X,pBest,pFitness,gBest,gFintness,new_pFitnes,new_beta,wBest)
    
#         MSE(iter) = gFitness
#             for (i=1 to Np):
#                 for (j=1 to Dim):
#                     z1(1,j) = c1 *rand *(pBest(i,j)-x(i,j))
#                     z2(1,j) = c2 *rand *(pBest(i,j)-x(i,j))
                
#         V=alpha(iter)*V+z1+z2
#         x=x+V
#     return (gBest ,wBest,MSE)

# def upd_particle(X,pBest,pFitness,gBest,gFitness,new_pFitnes,new_beta,wBest):
#     id_x = find (new_pFitnes< pFitness)
#     if (numel (id_x)>0):
#         pFitness (id_x) =new_pFitnes(id_x)
#         pBest (id_x,:) = x(id_x)
        
#     [new_gFitness, id_x ] = min (pFitness)
#     if (new_gFitness <  gFitness):
#         gFitness = new_gFitness
#         gBest = pBest(id_x,:)
#         wBest = new_beta(id_x,:)
        
#     return (pBest,pFitness,gBest,gFitness,wBest)

# def config_swarm(Np, Nh, D, MaxIter, inf):
#   X = ini_swarm(Np, Nh, D)
#   Dim = X.shape[1]
#   pBest = np.zeros((Np, Dim))
#   pFitness = np.ones((1, Dim)) * inf
#   gBest = np.zeros((1, Dim))
#   gFitness = inf
#   wBest = np.zeros((1, Nh))
#   Alpha = generateAlpha(MaxIter)
#   return X, pBest, pFitness, gBest, gFitness, wBest, Alpha



# def fitness(xe, X):
#   D, N = xe.shape
#   Np = X.shape[0]
#   for i in range(Np):
#     p = X[i]
#     w1 = np.reshape(p, Nh, D)
#     H = Activation(xe, w1)
#     W2[i] = mlp_pinv(H, ye, C)
#     ze = W2[i] * H
#     #MSE(i) = sqrt(mse(ye-ze))
#   return MSE, W2

# def loadParamConfig():
#   nh = 41
#   np = 41
#   maxIter = 10
#   c = 0.1
#   return nh, np, maxIter, c

# inf = 99999999999
# D, N = xe.shape
# Xe = np.ones((1, N))
# D, M = xv.shape
# Xv = np.ones((1, M))
# Nh, Np, maxIter, C = loadParamConfig()
# X, pBest, pFitness, gBest, gFitness, wBest, Alpha = config_swarm(Np, Nh, D, maxIter, inf)

# print(sqrt(4))
# print(random.randint(0, 1))
# print(matlib.repmat(1,2,3))
# print(xe.shape[0])
# print(transpose([[2,2,2],[3,3,3],[4,4,4]]))
# print(linalg.inv([[3,4,-1],[2,0,1],[1,3,-2]]))