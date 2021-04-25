import numpy as np

def sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(b - a*x))

def sigmoid_norm(x, a, b, left, right):
    return (sigmoid(x, a, b) - sigmoid(left, a, b)) / (sigmoid(right, a, b) - sigmoid(left, a, b))
