# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:28:43 2024

@author: Mattia
"""
import numpy as np
from layer import Layer
from utils import one_hot_encode

def cross_entropy_loss(outputs: np.ndarray, targets: np.ndarray) -> float:
    """
    Parameters
    ----------
    outputs: outputs of model of shape: (num_classes/num_neurons_output_layer, batch_size)
    targets: labels/targets of each image of shape: (num_classes, batch_size)
    
    Returns
    ----------
        Cross entropy error (float)
        
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    loss = - np.sum(targets * np.log(outputs), axis=0)
    
    #avg between the loss for each sample of the batch
    return np.average(loss)

def cross_entropy_loss_derivative(outputs: np.ndarray, targets: np.ndarray):
    """
    
    Parameters
    ----------
    outputs: outputs of model of shape: (num_classes, batch_size)
    targets: shape (num_classes, batch_size)
    
    Returns
    ----------
    derivatives shape (num_classes, batch_size)
        
    """
    
    return np.where(outputs != 0, -targets / outputs, 0.0)

def MSE(outputs: np.ndarray, targets: np.ndarray) -> float:
    """
    
    Parameters
    ----------
    outputs: outputs of model of shape: (number_of_outputs, batch_size)
    targets: labels/targets of each image of shape: (number_of_outputs, batch size)

    Returns
    ----------
    Mean squared error (float)
    
    """
    assert outputs.shape == targets.shape

    return np.average(np.average((outputs - targets)**2, axis=0))

def MSE_derivative(outputs, targets):
    """
    Parameters
    ----------
    outputs: np.ndarray of shape (number_of_outputs, batch_size)
    targets: np.ndarray of shape (number_of_outputs, batch size)
    Returns
    ----------
    np.ndarray of shape (number_of_outputs, batch_size)
    """

    number_of_outputs = outputs.shape[0]
    
    return 2.0 *(outputs - targets) / number_of_outputs

def L1_loss_contribute(weights, a):
    return np.sum(np.array(abs(weights))) * a
    
def L2_loss_contribute(weights, a):
    return np.sum(np.array(weights) ** 2) * a / 2

def run_loss_function(loss_function, outputs, targets, weights=[], reg_ratio=0, regularaizer="none"): 
    #regularaziation term to prevent too high weigths so too complexity and overfitting of the net
    if regularaizer == "L1": #promote scatter weights 
        regularaizer_term = L1_loss_contribute(weights, reg_ratio)
    if regularaizer == "L2": #promote reguaral spread of weights
        regularaizer_term = L2_loss_contribute(weights, reg_ratio) 
    else:
        regularaizer_term = 0       
        
    loss = 0
    
    if loss_function == "MSE":
        loss = MSE(outputs, targets)
    elif loss_function == "cross_entropy":
        loss = cross_entropy_loss(outputs, targets)
    else:
        raise NotImplementedError()

    return loss + regularaizer_term

def derivative_loss_function(loss_function, outputs, targets):
    if loss_function == "MSE":
        return MSE_derivative(outputs, targets)
    elif loss_function == "cross_entropy":
        return cross_entropy_loss_derivative(outputs, targets)
    else:
        raise NotImplementedError()

class NeuralNetwork:
    def __init__(self, learning_rate, layer_list, softmax, loss_function, 
                 regularaizer, regularaizer_factor, initial_weight_ranges, 
                 initial_bias_ranges, verbose):
        
        """
        Parameters
        ----------
        learning_rate: float
        layer_list: list of layer 'object' ex {'units': 50, 'activation': 'relu'}
        softmax: boolean
        loss_function: string - "MSE", "cross-entropy" are the possibilities
        regularaizer: string - "L1","L2","none" are the possibilities
        regularaizer_factor: float
        initial_weight_ranges: string or list of int
        initial_bias_ranges: list of int
        verbose: boolean
        
        Return
        ----------
        none
        """
        
        if softmax:
            layer_list.append({'units': layer_list[-1]['units'], 'activation': 'softmax'})

        layer_types = [1] * len(layer_list) #middle layer
        layer_types[0]=0 #input layer
        layer_types[-1]=2 #output layer
            
        self.softmax = softmax
        self.loss_function = loss_function
        
        self.regularaizer = regularaizer
        self.regularaizer_factor = regularaizer_factor
        
        self.verbose = verbose
        
        #layers = [None]
        layers = []
        
        prev_neuron_count = 0
        
        for i, l in enumerate(layer_list):
            neuron_count = l['units']
            activation_function = l['activation']
            
            layers.append(Layer(layer_types[i], neuron_count, prev_neuron_count, activation_function, 
                                initial_weight_ranges, initial_bias_ranges, learning_rate, regularaizer, regularaizer_factor))
            
            prev_neuron_count = neuron_count
            
        self.layers = layers
    
    def get_weights(self):
        """
        Returns
        -------
        weights : list of float
            the list of the weights of the network

        """
        weights = []
        for layer in self.layers:
            #print(weights)
            weights.extend(layer.get_weights())
        
        return weights
        
    def batch_loader(self, X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle=True):
        """
        Creates a batch generator over the whole dataset (X, Y) which returns a generator 
        iterating over all the batches.
        This function is called once each epoch.

        Parameters
        ----------
        X: Input data (numpy array).
        Y: Target labels (numpy array).
        batch_size: Size of each batch.
        shuffle: Whether to shuffle the data before creating batches.

        Return
        ----------
        A generator that yields batches of data in the form (batch_X, batch_Y).
        
        """
        num_samples = X.shape[0]

        if shuffle:
            indices = np.random.permutation(num_samples)
            X = X[indices]
            Y = Y[indices]

        num_batches = num_samples // batch_size #floor division

        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = (batch_num + 1) * batch_size
            batch_X = X[start_idx:end_idx]
            batch_Y = Y[start_idx:end_idx]

            yield batch_X, batch_Y
        
    def forward_pass(self, X):
        """
        Performs forward pass over a mini-batch.
        
        Returns
        --------
        array(units__last_layer, batch_size)
            return the output_layer result (one for each batch sample)
        """
        
        outputs = X
        
        for i, layer in enumerate(self.layers):
            if layer.type == 0:
                continue
            
            #the previus_layer output is the input of the current layer
            #the current layer will computed its output
            outputs = layer.forward_pass(outputs)

        return outputs
    
    def backward_pass(self, outputs, targets):
        """
        Performs backward pass over a mini-batch.

        Parameters
        ----------
        outputs: np.ndarray of shape (output_size, batch_size)
        targets: np.ndarray the same shape as outputs
        """
        assert outputs.shape == targets.shape

        # Jacobian loss
        #shape(batch, num_outputs)
        R = derivative_loss_function(self.loss_function, outputs, targets).T
        for i, layer in reversed(list(enumerate(self.layers))):
            # Skip input layer
            if layer.type != 0: #not input
                R = layer.backward_pass(R)
    
    def train(self, epochs, batch_size, X_train, Y_train, X_val, Y_val, shuffle=True):
        """
        Performs the training phase (forward pass + backward propagation) over a number of epochs.

        Parameters
        ----------
        epochs: int
        X_train: np.ndarray of shape (training dataset size, input_size)
        Y_train: np.ndarray of shape (training dataset size, output_size)
        X_val: np.ndarray of shape (validation dataset size, input_size)
        Y_val: np.ndarray of shape (validation dataset size, output_size)
        shuffle: bool, whether or not to shuffle the dataset
        
        """
        print("Training over {} epochs with a batch size of {}".format(
            epochs, batch_size))

        # Transpose X and Y because we want them as column vectors
        X_val = X_val.T
        Y_val = Y_val.T

        train_loss_history = []
        val_loss_history = []

        iteration = 0
       
        for epoch in range(epochs):
            
            train_loader = self.batch_loader(
                X_train, Y_train, batch_size, shuffle=shuffle)
            
            for X_batch, Y_batch in iter(train_loader):
                # Transpose X and Y because we want them as column vectors
                X_batch = X_batch.T
                Y_batch = Y_batch.T
                
                output_train = self.forward_pass(X_batch)
                
                self.backward_pass(output_train, Y_batch)
                
                if self.regularaizer == "L1" or self.regularaizer == "L2":
                    loss_train = run_loss_function(
                        self.loss_function, output_train, Y_batch,
                        self.get_weights(), self.regularaizer_factor, self.regularaizer)
                    
                else:
                    loss_train = run_loss_function(
                        self.loss_function, output_train, Y_batch)
                    

                train_loss_history.append(loss_train)

                output_val = self.forward_pass(X_val)
                
                if self.regularaizer == "L1" or self.regularaizer == "L2":
                    loss_val = run_loss_function(
                        self.loss_function, output_val, Y_val,
                        self.get_weights(), self.regularaizer_factor, self.regularaizer
                    )
                else:
                    loss_val = run_loss_function(
                        self.loss_function, output_val, Y_val,
                    )

                
                val_loss_history.append(loss_val)

                if self.verbose == True:
                    print("-epoch:{}, iter:{}, train loss: {:.4f}, val loss {:.4f}".format(
                        epoch, iteration, loss_train, loss_val))

                iteration += 1

        return train_loss_history, val_loss_history
    
    def predict(self, X, one_hot=True):
        """
        Feeds X into the model and returns the output.

        Parameters
        ----------
        X: np.ndarray of shape (batch_size, input_size)
        one_hot: bool, whether the output should be one-hot encoded.

        Return
        ----------
        np.ndarray of shape (batch_size, output_size)
        """
        
        output = self.forward_pass(X.T)
        if one_hot:
            output = one_hot_encode(output)
        return output.T
    

    
    