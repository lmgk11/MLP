import MLP
import ACTIVATION
import numpy as np 
import random
import pickle

inp = [
        np.array([[1], [0]]),
        np.array([[0], [1]]),
        np.array([[1], [1]]),
        np.array([[0], [0]])
        ]

des = [
        np.array([[1]]),
        np.array([[1]]),
        np.array([[0]]),
        np.array([[0]])
        ]

m = MLP.DNN('2x4x4x4x4x1')
#m.load_net('nn_xor.ps')

ITERATIONS = 10_000

for j in range(100):
    correct = 0
    for i in range(ITERATIONS):
        index = random.randrange(0,4)
        ans = m.backpropagate(inp[index], des[index], True)
        if (ans >= 0.5 and index in [0,1]) or (ans < 0.5 and index in [2,3]):
            correct = correct + 1 
    print(f'After {(j+1)*ITERATIONS} Iterations: {100 * correct / ITERATIONS}%')
    if correct / ITERATIONS >= 0.99:
        break





print(f' INPUT: 1 0 => {round(m.feed_forward(inp[0])[0][0])}')
print(f' INPUT: 0 1 => {round(m.feed_forward(inp[1])[0][0])}')
print(f' INPUT: 1 1 => {round(m.feed_forward(inp[2])[0][0])}')
print(f' INPUT: 0 0 => {round(m.feed_forward(inp[3])[0][0])}')

m.save_net_to_file('nn_xor.ps')
