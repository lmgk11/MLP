import numpy as np
import random
import MLP
import os

from keras.datasets import mnist
from tqdm import tqdm

def classify_desired(y):
    des = np.zeros((10, 1))
    des[y] = 1
    return des

def classify_result(y):
    return y.flatten().tolist().index(max(y.flatten().tolist()))

m = MLP.DNN('784x2500x2000x1500x1000x500x10', 0.01)
#m.load_net('mnist_3.ps')
(train_X, train_y), (test_X, test_y) = mnist.load_data()

os.system('clear')

print(f'Starting training with learning rate: {m.learning_rate} \n\n')


while True:

    correct = 0
    for i in tqdm(range(5000), ncols = 80, desc='Training net'):
        rand_index = random.randrange(60_000)
        m.backpropagate(np.reshape(train_X[rand_index].flatten() * (1/255), (-1, 1)), classify_desired(train_y[rand_index]), False)

    m.save_net_to_file('mnist_dnn.ps')
    print('\nTrained on 5000 images, weights saved, now testing...\n')

    for q in tqdm(range(10_000), ncols = 80,desc='Testing net'):
        if classify_result(m.feed_forward(np.reshape(train_X[q].flatten() * (1/255), (-1, 1)))) == train_y[q]:
            correct = correct + 1
    #m.set_learning_rate(m.learning_rate * 0.5 )
    print(f'\nTested 10,000 images: RESULT {100 * correct / 10000}%')
    



