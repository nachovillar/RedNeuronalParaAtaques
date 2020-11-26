#Importamos la data como matriz y como frame
from train_input import dataMatrix
from functions import activation
# from param_config import Xe, Ye, L, C
from numpy import random, shape, sqrt, matlib, transpose
# Librería para acceder a la inversa de las matrices
from scipy import linalg
#Matriz con los datos de entrada
xe = dataMatrix
#Dimensión de los datos de entrada
sizeXe = xe.shape
# FALTA CARGAR YE, L y C!!!!
# [w1, bias, w2] = upd_pesos(xeMatrix, Ye, L, C)
# costo = calc_costo(xeMatrix)




#params: Data de entrada, Data salida, Nodos ocultos, Penalidad pseudo inversa
def upd_pesos (Xe, Ye, L, C):
    [nodos_entrada, caracteristicas] = sizeXe
    w1 = rand_W(L, nodos_entrada) #Peso Oculto del último nodo de entrada hacia el último nodo oculto I GUESS
    bias = rand_W(L, 1) #Sesgo--> peso oculto del primer nodo de entrada hacia el último nodo oculto I GUESS
    biasMatrix = matlib.repmat(bias, nodos_entrada, caracteristicas) #Matriz de pesos ocultos

    z = w1*xe + biasMatrix
    H = activation(z)
    Htrnaspose = transpose(H)
    #Se calculan los pesos de salida
    yh = Ye*Htrnaspose
    hh = (H*Htrnaspose + eye(L)/C)

    inverse = linalg.inv(hh)
    w2 = yh*inverse

    return(w1, bias, w2) 

def calc_costo(Xe, Ye, w1, bias, w2):
    return
#Función para inicializar los pesos
def rand_W(next_nodes, current_nodes):
    r = sqrt(6/(next_nodes + current_nodes))
    w = (random.randint(next_nodes, current_nodes))*2*r - r
    return(w)

def PSO():
    for (iter=1 to maxIter):
        new_pFitness, new_beta = fitness (Xe,Ye,Nh,X,Cpinv)
        pBest,pFitness,gBest,gFintness,wBest =upd_particle(X,pBest,pFitness,gBest,gFintness,new_pFitnes,new_beta,wBest)
    
        MSE(iter) = gFitness
            for (i=1 to Np):
                for (j=1 to Dim):
                    z1(1,j) = c1 *rand *(pBest(i,j)-x(i,j))
                    z2(1,j) = c2 *rand *(pBest(i,j)-x(i,j))
                
        V=alpha(iter)*V+z1+z2
        x=x+V
    return (gBest ,wBest,MSE)

def upd_particle(X,pBest,pFitness,gBest,gFitness,new_pFitnes,new_beta,wBest):
    id_x = find (new_pFitnes< pFitness)
    if (numel (id_x)>0):
        pFitness (id_x) =new_pFitnes(id_x)
        pBest (id_x,:) = x(id_x)
        
    [new_gFitness, id_x ] = min (pFitness)
    if (new_gFitness <  gFitness):
        gFitness = new_gFitness
        gBest = pBest(id_x,:)
        wBest = new_beta(id_x,:)
        
    return (pBest,pFitness,gBest,gFitness,wBest)

def fitness():
    [D N] = size (Xe)
    Np = size (X,1)
    
    for (i=1 to Np):
        p = 
    


print(sqrt(4))
print(random.randint(0, 1))
print(matlib.repmat(1,2,3))
print(xe.shape[0])
print(transpose([[2,2,2],[3,3,3],[4,4,4]]))
print(linalg.inv([[3,4,-1],[2,0,1],[1,3,-2]]))