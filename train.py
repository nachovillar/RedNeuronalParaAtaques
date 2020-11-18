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


print(sqrt(4))
print(random.randint(0, 1))
print(matlib.repmat(1,2,3))
print(xe.shape[0])
print(transpose([[2,2,2],[3,3,3],[4,4,4]]))
print(linalg.inv([[3,4,-1],[2,0,1],[1,3,-2]]))