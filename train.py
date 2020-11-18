from train_input import dataMatrix, dataFrame
# from param_config import Xe, Ye, L, C
from numpy import random, repmat


dim = dataFrame.shape[0]
print(dataMatrix)
print("-------------")
print(dataFrame.shape[0])


def upd_pesos (Xe, Ye, L, C):
    [row, col] = dataFrame.shape
    w1 = random(L, dimension)
    bias = random(L, 1)
    biasMatrix = repmat(bias, 1, row)

