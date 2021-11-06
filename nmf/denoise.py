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


def get_speech_signal(V_mixed, B_speech, B_noise, n_iter):
    # first learn W
    assert(B_speech.shape == B_noise.shape)
    p, k = B_speech.shape

    B = np.concatenate([B_speech, B_noise], axis=1)
    assert(B.shape == (p, 2 * k))

    p, q = V_mixed.shape
    W = np.random.rand(2 * k, q)

    for _ in range(n_iter):
        W_numerator = B.T @ np.divide(V_mixed, B @ W)
        W_denominator = B.T @ np.ones((p, q))
        W_div = np.divide(W_numerator, W_denominator)
        W = np.multiply(W, W_div)

    assert(W.shape == (2 * k, q))

    # reconstruction with learned H
    W_speech = W[:k, :]

    V_speech_rec = B_speech @ W_speech

    return V_speech_rec
