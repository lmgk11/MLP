import numpy as np
import random
import MLP
import os

from keras.datasets import mnist
from tqdm import tqdm

FILENAME = 'mnist_dnn_784_800_10.ps'

def classify_desired(y):
    des = np.zeros((10, 1))
    des[y] = 1
    return des

def classify_result(y):
    return y.flatten().tolist().index(max(y.flatten().tolist()))

m = MLP.DNN('784x400x400x200x10', 0.0001, 0.005)
m.load_net(FILENAME)
(train_X, train_y), (test_X, test_y) = mnist.load_data()

os.system('clear')


print('Testing net for reference...')

correct = 0

for j in range(10_000):
    if classify_result(m.feed_forward(np.reshape(train_X[j].flatten() * (1/255), (-1, 1)))) == train_y[j]:
        correct = correct + 1

print(f'\nReference: {100 * correct / 10_000}%\n')

MAXIMUM = 100 * correct / 10_000 

print(f'Starting training with learning rate: {m.learning_rate} \n')

while True:

    correct = 0
    train_correct = 0
    error = False
    for i in tqdm(range(60_000), ncols = 80, desc='Training net'):
        #6rand_index = random.randrange(60_000)
        rand_index = i
        
        ans = m.backpropagate(np.reshape(train_X[rand_index].flatten() * (1/255), (-1, 1)), classify_desired(train_y[rand_index]), True)
            
        if classify_result(ans) == train_y[rand_index]:
            train_correct += 1 

        


    print(f'\nTrained on 5000 images with accuracy {train_correct / 60_000}, now testing...\n')

    for q in tqdm(range(10_000), ncols = 80,desc='Testing net'):
        if classify_result(m.feed_forward(np.reshape(train_X[q].flatten() * (1/255), (-1, 1)))) == train_y[q]:
            correct = correct + 1
    if 100 * correct / 10_000 > MAXIMUM:
        MAXIMUM = 100 * correct / 10_000
        m.save_net_to_file(FILENAME)

        print(f'NEW HIGHEST RESULT: {MAXIMUM / 100} \n\nSaving net to file...')
    #m.set_learning_rate(m.learning_rate * 0.5 )
    else:
        print(f'\nTested 10,000 images: RESULT {100 * correct / 10000}%\n')
    



