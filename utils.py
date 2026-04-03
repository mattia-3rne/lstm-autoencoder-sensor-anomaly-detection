import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x)**2