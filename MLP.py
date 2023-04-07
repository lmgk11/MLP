import numpy as np 
import pickle

import ACTIVATION


class MLP:
    def __init__(self, input_neurons, hidden_neurons, output_neurons, learning_rate=0.001):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        self.weights_input_hidden = np.random.rand(self.hidden_neurons, self.input_neurons)
        self.weights_hidden_output = np.random.rand(self.output_neurons, self.hidden_neurons)
        
        self.hidden_bias = np.random.rand(self.hidden_neurons, 1)
        self.output_bias = np.random.rand(self.output_neurons, 1) 

        self.activation_function, self.activation_dfunc = ACTIVATION.sigmoid()

        self.learning_rate = learning_rate

    def load_net(self, FILENAME):
        with open(FILENAME, 'rb') as file:
            w = pickle.load(file)
            self.weights_input_hidden = w[0]
            self.weights_hidden_output = w[1]
            self.hidden_bias = w[2]
            self.output_bias = w[3]

        self.activation_function, self.activation_dfunc = ACTIVATION.sigmoid()

    def feed_forward(self, input):
        hidden = self.activation_function(np.matmul(self.weights_input_hidden, input) + self.hidden_bias)
        return self.activation_function(np.matmul(self.weights_hidden_output, hidden) + self.output_bias)
        
    def backpropagate(self, input, desired, return_output):
        hidden = self.activation_function(np.matmul(self.weights_input_hidden, input) + self.hidden_bias)
        output = self.activation_function(np.matmul(self.weights_hidden_output, hidden) + self.output_bias)
        try: 
            assert output.shape == desired.shape
        except AssertionError:
            print(f'Desired output is of wrong shape...\nExpected {output.shape} \nExiting...')
            exit()

        output_error = desired - output 
        gradient = self.activation_dfunc(output)
        gradient = self.learning_rate * np.multiply(gradient, output_error)
        
        self.weights_hidden_output = self.weights_hidden_output + np.matmul(gradient, np.transpose(hidden))
        self.output_bias = self.output_bias + gradient 

        hidden_error = np.matmul(np.transpose(self.weights_hidden_output), output_error)
        hidden_gradient = self.learning_rate * np.multiply(self.activation_dfunc(hidden), hidden_error)

        self.weights_input_hidden = self.weights_input_hidden + np.matmul(hidden_gradient, np.transpose(input))
        self.hidden_bias = self.hidden_bias + hidden_gradient
        if return_output:
            return output


    def save_net_to_file(self, FILENAME):
        with open(FILENAME, 'wb') as file:
            pickle.dump([self.weights_input_hidden, self.weights_hidden_output, self.hidden_bias, self.output_bias], file)

    def set_activation(self, func):
        self.activation_function, self.activation_dfunc = func
    
    def print_weights(self):
        print(self.weights_input_hidden, self.weights_hidden_output, sep='\n\n')

    def set_learning_rate(self, lr):
        self.learning_rate = lr

class DNN:
    def __init__(self, architecture, LR=0.01):
        self.weights = []
        self.bias = []
        self.nodes = [int(node) for node in architecture.split('x')]
        self.learning_rate = LR

        for i in range(len(self.nodes) - 1):
            self.weights.append(np.random.rand(self.nodes[i+1], self.nodes[i]))
            self.bias.append(np.random.rand(self.nodes[i+1], 1))
        self.activation_function, self.activation_dfunc = ACTIVATION.sigmoid()

    def feed_forward(self, input, predict=True):
        self.activation_function, self.activation_dfunc = ACTIVATION.relu()
        hidden = [input]
        for i, weight in enumerate(self.weights):
            if i == len(self.weights) - 1:
                self.activation_function, self.activation_dfunc = ACTIVATION.sigmoid()
            hidden.append(self.activation_function(np.matmul(weight, hidden[i]) + self.bias[i]))

        
        return hidden[-1] if predict else hidden

    def backpropagate(self, inputs, desired, return_output = False):
        layers = self.feed_forward(inputs, False)
        
        try:
            assert desired.shape == layers[-1].shape 
        except AssertionError:
            print('Desired output is of wrong shape.\nExiting...')
            exit()
        
        #self.activation_function, self.activation_dfunc = ACTIVATION.sigmoid()
        
        error = desired - layers[-1]

        for i in reversed(list(range(len(layers) - 1))):
            gradient = self.learning_rate * np.multiply(self.activation_dfunc(layers[i+1]), error)
            
            self.weights[i] += np.matmul(gradient, np.transpose(layers[i]))
            self.bias[i] += gradient 

            error = np.matmul(np.transpose(self.weights[i]), error)
            
            self.activation_function, self.activation_dfunc = ACTIVATION.relu()
        
        self.activation_function, self.activation_dfunc = ACTIVATION.sigmoid()

        if return_output:
            return layers[-1]

    def load_net(self, FILENAME):
        with open(FILENAME, 'rb') as file:
            self.weights = pickle.load(file)

    def save_net_to_file(self, FILENAME):
        with open(FILENAME, 'wb') as file:
            pickle.dump(self.weights, file)

    def print_weights(self):
        print(self.weights)
