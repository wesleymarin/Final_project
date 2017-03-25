import numpy as np
import math
import time
import matplotlib.pyplot as plt
import itertools as it

class NeuralNetwork():
    """
    This object does all of the neural network computations
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        When an NN object is first created, it needs to be initialized for
        input layer size, hidden layer size, and output layer size. all
        other parameters are set here as well, and can be changed as needed.

        Input:  Int for input layer
                Int for hidden layer size
                Int for output layer size
        Output: NeuralNetwork object
        """

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lam = 0.001
        self.z1 = np.zeros(hidden_size)
        self.z2 = np.zeros(output_size)
        self.alpha = 1
        self.cost_history = []

        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)

        self.hidden_layer = np.zeros(hidden_size)
        self.output_layer = np.zeros(output_size)

        self.b1 = np.random.randn(hidden_size)
        self.b2 = np.random.randn(output_size)

    def sigmoid_derivative(self, x):
        """
        This function is the derivative of the sigmoid function
        """
        return x * (1 - x)

    def sigmoid(self, x):
        """
        This is a sigmoid function
        """
        f_x = 1 / (1 + np.exp(-x))
        return(f_x)

    def cost_function(self):
        """
        This is a cost function that I had great plans for, but never actually used.
        Cost prime is the one I do use.
        """
        self.forward()
        avg_ss = 0.5*sum((self.true_layer - self.output_layer)**2)
        J = sum(avg_ss) / self.output_size
        return J

    def cost_prime(self, layer):
        """
        This is the derivative of the cost function

        Input:  A layer to compute the cost derivative of
        Output: A cost derivative array
        """
        dJ = np.multiply(layer, 1-layer)
        return dJ

    def activation(self, input_vector, weights, bias):
        """
        This computes the sigmoid activation of the input layer based on the
        input, weights and bias.

        Input:  Layer to compute the activation of
                Weight array
                Bias array
        Output: The pre activated array
                The post activated array
        """
        ## Compute Z by multiplying the input layer and weights, then adding bias
        z = input_vector.dot(weights) + bias

        ## Compute a by passing z through the sigmoid
        a = self.sigmoid(z)
        return(z, a)

    def compute_grad(self):
        """
        This function computes the gradient

        Output: del of b1
                del of b2
                dJ/dW1
                dJ/dW2
        """
        ## Figure out the error of predictions vs actual
        error_vector = -(self.true_layer - self.output_layer)

        ## Compute gradient, lots of array multiplication and dot products
        del_output = np.multiply(error_vector, self.cost_prime(self.output_layer))
        del_hidden = np.multiply(del_output.dot(self.W2.T), self.cost_prime(self.hidden_layer))

        dJ_dW2 = self.hidden_layer.T.dot(del_output)
        del_b2 = np.sum(del_output, axis = 0)
        del_b2 = del_b2 / self.input_layer.shape[0]

        dJ_dW1 = self.input_layer.T.dot(del_hidden)
        del_b1 = np.sum(del_hidden, axis = 0)
        del_b1 = del_b1 / self.input_layer.shape[0]

        return(del_b1, del_b2, dJ_dW1, dJ_dW2)

    def forward(self):
        """
        The feed-forward function.
        """
        ## Input layer to hidden layer
        self.z1, self.hidden_layer = self.activation(self.input_layer, self.W1, self.b1)

        ## Hidden layer to output layer
        self.z2, self.output_layer = self.activation(self.hidden_layer, self.W2, self.b2)


    def train(self, input_vector, true_vector, iterations):
        """
        The training function, takes the input layer, feeds it forward, fires
        the gradient determination function, then updates the weights and bias
        arrays.

        Input:  Input layer
                True values
                How many times to train
        """
        ## Turn the input and true into object properties
        self.input_layer = input_vector
        self.true_layer = true_vector

        for i in range(0, iterations):
            ## Fire the feed forward function
            self.forward()

            ## Fire the gradient determination function
            del_b1, del_b2, dJ_dW1, dJ_dW2 = self.compute_grad()

            ## Doing some normalization and scaling
            dJ_dW2 = dJ_dW2 / input_vector.shape[0] + self.lam * self.W2
            del_b2 = del_b2 / input_vector.shape[0]

            dJ_dW1 = dJ_dW1 / input_vector.shape[0] + self.lam * self.W1
            del_b1 = del_b1 / input_vector.shape[0]

            ## Updating weights and bias according to alpha scaling
            self.W2 = self.W2 - self.alpha * dJ_dW2
            self.b2 = self.b2 - self.alpha * del_b2

            self.W1 = self.W1 - self.alpha * dJ_dW1
            self.b1 = self.b1 - self.alpha * del_b1

    def predict(self, input_vector):
        """
        This function takes an input layer and feeds it forward.

        Input:  Input layer
        Output: Predictions (output layer)
        """
        self.input_layer = input_vector
        self.forward()
        return(self.output_layer)

"""
My input method relies on groups of 3, so I enumerate all possible combinations
of the three DNA bases, and use those as dictionary keys. Afterwards, each
key gets its own binary vector. The window size can be varied, but I did
not do that.
"""
codons = it.product('ATGC', repeat=3)
dna_conv_dict = {}
window_keys = [x for x in codons]

## Intializing all of the values at once by creating a massive array
window_values = np.zeros((len(window_keys), len(window_keys)), dtype=np.int)

## Add a one to a different position in each value, then pair that value
## with a key. Every key and every value are unique, it doesn't matter which
## two are paired together.
for i in range(0,window_values.shape[0]):
    window_values[i][i] = 1
    dna_conv_dict[window_keys[i]] = window_values[i]

## This is used by the reverse complement function
comp_dict = {'A':'T', 'T':'A', 'G':'C', 'C':'G'}

class DNAsequence():
    """
    This object turns a DNA sequene string into a number array based on the
    dna_conv_dict created previously.
    """
    def __init__(self, sequence, region_size):
        """
        This could have been a function instead of an object, but i thought
        more flexibility would be nice.

        This is all based on http://file.scirp.org/Html/3-9102277_65923.htm

        Input:  Sequence string
                Size of the region
        """
        ## save the letter sequence to the object
        self.letter_sequence = sequence

        ## Initialize the number array
        self.num_of_regions = len(sequence)-2-(region_size-1)
        self.number_array = np.zeros((self.num_of_regions,region_size*64))

        ## Convert the sequences into numbers and put them into the array
        for i in range(0,self.num_of_regions):
            for j in range(0,region_size):
                self.number_array[i,j*64:(j+1)*64] = dna_conv_dict[tuple(self.letter_sequence[0+i+j:3+i+j])]
