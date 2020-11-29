# Importamos la data como matriz y como frame
from train_input import ye, xe
from test_input import yv, xv
# Librería para acceder a la inversa de las matrices
from scipy import linalg
from PSO import *
import pandas as pd


def activation(Xe, W1):
    z = W1*Xe

    return ((2 / (1 + exp(-z))) - 1)


# Función para inicializar los pesos
def rand_W(next_nodes, current_nodes):
    r = sqrt(6 / (next_nodes + current_nodes))
    w = random.rand(next_nodes, current_nodes)
    w = (w * 2 * r) - r
    return w


# params: Data de entrada, Data salida, Nodos ocultos, Penalidad pseudo inversa
def upd_pesos(Xe, Ye, L, C):
    [nodos_entrada, caracteristicas] = Xe.shape  # numero de nodos de entrada, atributos del nodo de entrada
    w1 = rand_W(L, nodos_entrada)  # Matriz de pesos ocultos
    bias = rand_W(L, 1)  # Sesgo
    biasMatrix = matlib.repmat(bias, 1, caracteristicas)  # Matriz para sumarle el sesgo a todos los pesos
    print(Ye)
    z = w1 * Xe + biasMatrix
    H = activation(z)
    Htranspose = transpose(H)
    # Se calculan los pesos de salida
    yh = Ye * Htranspose
    hh = (H * Htranspose + eye(L) / C)

    inverse = linalg.inv(hh)
    w2 = yh * inverse  # Pesos de Salida
    # Retornamos los pesos ocultos, sesgo y pesos de salida
    return [w1, bias, w2]

D, N = xe.shape
Xe = np.hstack(xe, np.ones(1, N))
D, M = xv.shape
Xv = np.hstack(xv, np.ones(1, M))

pso = PSO(D+1, Xe, ye)
gBest, wBest, MSE = pso.PSO()

gBest.to_csv('data/gBest.txt', sep=',')
wBest.to_csv('data/wBest.txt', sep=',')
MSE.to_csv('data/MSE.txt', sep=',')


