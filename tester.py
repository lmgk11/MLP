import MLP
import numpy as np
a = MLP.DNN('2x4x4')
a.backpropagate(np.random.rand(2,1), np.random.rand(4,1))
