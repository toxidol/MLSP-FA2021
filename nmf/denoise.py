import sys
import os
p = os.path.abspath('..')
sys.path.append(p)

from utils import *


def NMF_train(M, B_init, W_init, n_iter):
    B = B_init
    W = W_init

    p, q = M.shape

    for _ in range(n_iter):
        B_numerator = np.divide(M, B @ W) @ W.T
        B_denominator = np.ones((p, q)) @ W.T
        B_div = np.divide(B_numerator, B_denominator)
        B = np.multiply(B, B_div)

        W_numerator = B.T @ np.divide(M, B @ W)
        W_denominator = B.T @ np.ones((p, q))
        W_div = np.divide(W_numerator, W_denominator)
        W = np.multiply(W, W_div)

    assert(B.shape == B_init.shape)
    assert(W.shape == W_init.shape)

    return B, W
