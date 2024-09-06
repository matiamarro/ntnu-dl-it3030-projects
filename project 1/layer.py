# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:34:17 2024

@author: Mattia
"""
import numpy as np
    
def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtracting np.max(x) for numerical stability
                                 #
    return e_x / e_x.sum(axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def linear(x):
    return x

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - np.power(np.tanh(x), 2)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    x[x > 0] = 1
    x[x <= 0] = 0
    return x

def linear_derivative(x: np.ndarray) -> np.ndarray:
    return np.ones(x.shape)

#[softmax, sigmoid, tanh, relu, linear]
def activation_function(function, x):
    if function == "softmax":
        return softmax(x)
    elif function == "sigmoid":
        return sigmoid(x)
    elif function == "tanh":
        return tanh(x)
    elif function == "relu":
        return relu(x)
    elif function == "linear":
        return linear(x)
    else: 
        raise NotImplementedError()
        
def derivate_activation_function(function, x):
    if(function == "sigmoid"):
        return sigmoid_derivative(x)
    elif(function == "tanh"):
        return tanh_derivative(x)
    elif(function == "relu"):
        return relu_derivative(x)
    elif(function == "linear"):
        return linear_derivative(x)
    else:
        raise NotImplementedError()

def glorot_init(fan_in, fan_out): 
    # (input_units, output_units)
    # return a matrix of weights that are randomly picked in a uniformly way
    # in the range 'limit'
    
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=((fan_in, fan_out)))

class Layer:
    def __init__(self, layer_type, units, previous_units, activation_f, weights_range,
                 bias_range, learning_rate, regularaizer="none", regularaizer_factor = 0):
        """
        
        Parameters
        ----------
        layer_type : int
            {0: input, 1: middle, 2: output}  what type of layer is it
        units : int
            how many neuron units in the actual layer
        previous_units : int
            how many neuron units in the previus layer
        activation_f : String
            {sigmoid, tanh, relu, linear, softmax} are the considered option 
            and define which activation function is used
        weights_range : String of 2-lenght int list
            {glorot} or ex. {[0,1]} settings for first initialization of weights
        bias_range : 2-lenght int list
            settings for first initialization of biases
        learning_rate : float
        regularaizer : String, optional
            {L1, L2} are the useful settings or wathever means no regularizer. 
        regularaizer_factor : float, optional
            The default is 0.

        Returns
        -------
        None.

        """
        
        self.type = layer_type #[0: input, 1:hidden, 2:output]
        self.units = units # numer of neurons
        self.previous_units = previous_units # number of neurons for the previus layer
        self.activation_f = activation_f # activation function
        self.learning_rate = learning_rate
        self.weights = []
        self.regularaizer_factor = regularaizer_factor
        self.regularaizer = regularaizer
        
        if activation_f != "softmax" and layer_type != 0:
            self.bias = np.random.uniform(bias_range[0], 
                                             bias_range[1],
                                             size=(1, units))
            
            if weights_range == "glorot":
                self.weights = glorot_init(
                    previous_units, units)
            else:
                self.weights = np.random.uniform(weights_range[0], 
                                                 weights_range[1],
                                                 size=(previous_units, units))
    def get_weights(self):
        """
        
        Returns the list of weights of the entire network ex. [w1,w2,..wn]
        
        """
        
        if self.weights != []:
            return self.weights.flatten().tolist()
        return []
    
    def forward_pass(self, X):
        """
        Parameters
        ----------
            X: np.ndarray of shape (self.previous_units, batch_size)
        Returns
        ----------
            np.ndarray of shape (self.units, batch_size)
        """
        
        #shape(previous_units, batch)
        self.inputs = X        
                
        wi = X
        
        if self.activation_f != "softmax":
            wi = np.matmul(self.weights.T, X) + self.bias.T
        
        self.output = activation_function(self.activation_f, wi)
        
        #shape(units, batch)
        return self.output 
    
    def backward_pass(self, R):
        """
        Parameters
        ----------
            R: np.ndarray of shape (batch_size, units)

        Returns
        ----------
            np.ndarray of shape (batch_size, previous_units)
        """
        if self.activation_f == "softmax":
            output = self.output.T  # (batch_size, neurons)

            batch_size = output.shape[0]
            for b in range(batch_size):
                #derivate of softmax is a function, so for each sample
                #we get a matrix and from that 
                jacobian_matrix  = np.empty((self.units, self.units))
                for i in range(self.units):
                    for j in range(self.units):
                        if i == j:
                            jacobian_matrix [i, j] = output[b, i] - (output[b, i] ** 2)
                        else:
                            jacobian_matrix [i, j] = - output[b, i] * output[b, j]

                R[b] = np.matmul(R[b], jacobian_matrix)

            #(batch_size, units)
            return R
        
        #[.. , ..] = [.. , ..] * [.. , ..]      (for 1 sample)
        #shape(batch, units)
        delta = R * derivate_activation_function(self.activation_f, self.output).T 
        
        #will be computed gradient of weights and biases for 
        #aech sample of the batch. The upgrading of the parameters
        #will be influenced by each of them perfrorming an avg
        batch_size = R.shape[0]
        #shape(previus_units, units)
        gradient_weights = np.matmul(self.inputs, delta) / batch_size
        #shape(1, units)           .T -> shape(units,1)
        gradient_bias = delta.sum(axis=0, keepdims=True) / batch_size
        
        self.weights -= self.learning_rate * gradient_weights
        #L2
        if self.regularaizer == "L2":
            self.weights -= self.learning_rate * self.regularaizer_factor * self.weights
        #L1
        elif self.regularaizer == "L1":
            self.weights -= self.learning_rate * self.regularaizer_factor * np.sign(self.weights)
        
        self.bias -= self.learning_rate * gradient_bias
    
        #[n] x [n][m] = [m]     (for 1 sample)
        #n = number units in current layer
        #m = number units in previus layer (each current neuron get one weight from
        #                                   each previus neuron)
        #shape(batch, previous_units)
        return np.matmul(delta, self.weights.T)
        
        
        
        
        
        