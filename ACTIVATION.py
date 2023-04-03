import math
import numpy as np

def sigmoid():
    return np.vectorize(lambda x: (1 / (1 + math.exp(-x)))), np.vectorize(lambda x: (math.exp(-x) / ((1+math.exp(-x))**2)))

def tanh():
    return np.vectorize(lambda x: math.tanh(x)), np.vectorize(lambda x: 1 - math.tanh(x)**2)

