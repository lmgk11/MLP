import matplotlib.pyplot as plt 
import random
import numpy as np

import MLP

from keras.datasets import mnist

def classify_result(y):
    return y.flatten().tolist().index(max(y.flatten().tolist()))



m = MLP.MLP(784, 800, 10)

(train_X, train_y), (test_X, test_y) = mnist.load_data()


sample = random.randrange(10_000)

image = test_X[sample]

print('\n\n\n')

m.load_net('mnist.ps')


fig = plt.figure
plt.imshow(image, cmap='gray')
plt.title(f'I think this is a {classify_result(m.feed_forward(np.reshape(test_X[sample].flatten() * (1/255), (-1, 1))))}', fontsize=40)
plt.show()

print(f'Correct answer was a {test_y[sample]}')
