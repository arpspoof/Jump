import numpy as np

def bell(x, target, half_tolerance):
    k = np.log(2) / (half_tolerance**2)
    return np.exp(-k*(x - target)**2)
