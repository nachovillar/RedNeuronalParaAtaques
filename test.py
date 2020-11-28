from numpy import matrix, shape, matlib
import numpy as np
# Librería para el manejo de archivos .csv
import pandas as pd
from test_input import yv, xv

def forward(xv, w1, bias, w2):
    [D, N] = xv.shape
    biasMatrix = matlib.repmat(bias, 1, N)
    z = w1*xv + biasMatrix
    H = activation(z)
    z = w2*H

    return(z)

z = forward(xv, w1, bias, w2)
[accuracy, Fscore] = metrica(z, yv)



