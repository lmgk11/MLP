import numpy as np
import random
import MLP

from keras.datasets import mnist

def classify_desired(y):
    des = np.zeros((10, 1))
    des[y] = 1
    return des

def classify_result(y):
    return y.flatten().tolist().index(max(y.flatten().tolist()))

m = MLP.MLP(784, 800, 10, 0.001)
m.load_net('mnist.ps')
m.set_learning_rate(0.00001)
(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('\n\n\n')

for j in range(5):
    correct = 0
    for i in range(1000):
        rand_index = random.randrange(60_000)
        ans = m.backpropagate(np.reshape(train_X[rand_index].flatten() * (1/255), (-1, 1)), classify_desired(train_y[rand_index]), True)
        if train_y[rand_index] == classify_result(ans):
            correct = correct + 1 
    r = random.randrange(60_000)
    t = m.feed_forward(np.reshape(train_X[r].flatten() * (1/255), (-1, 1)))
    print(f'CORRECT {100 * correct / 1000}% \t TEST: GUESS = {classify_result(t)} \t CORRECT = {train_y[r]}')
    


m.save_net_to_file('mnist.ps')
