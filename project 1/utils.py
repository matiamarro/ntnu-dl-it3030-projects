# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:32:37 2024

@author: Mattia
"""
import numpy as np
import matplotlib.pyplot as plt
       
def accuracy(outputs, targets):
    """
    Calculates the accuracy (i.e., correct predictions / total predictions).

    Args:
        outputs: np.ndarray of shape (number_of_outputs, batch_size)
        targets: same dimension as outputs
    Return
        accuracy: float
    """
    assert outputs.shape == targets.shape, "Shape was {}, {}".format(
        outputs.shape, targets.shape)

    outputs = one_hot_encode(outputs)
    dataset_size = outputs.shape[1]
    correct_predictions = 0
    for i in range(dataset_size):
        if np.all(np.equal(outputs[:, i], targets[:, i])):
            correct_predictions += 1

    return correct_predictions / dataset_size

def one_hot_encode(X):
    """
    One-hot encodes the input

    Args
        X: np.ndarray of shape (number of outputs, batch_size)

    Return
        one-hot encoded version of X
    """
    output = np.zeros_like(X)
    max_indexes = np.argmax(X, axis=0)
    output[max_indexes, np.arange(X.shape[1])] = 1
    return output

def split_dataset(X, Y, training_ratio=0.7, validation_ratio=0.2, test_ratio=0.1, seed=None):
    """
    Splits the dataset into training, validation, and testing data.

    Args:
        X: inputs of shape (dataset_size, input_size)
        Y: targets of shape (dataset_size, output_size)
        training_ratio: Ratio of data to be used for training (default is 0.7)
        validation_ratio: Ratio of data to be used for validation (default is 0.2)
        test_ratio: Ratio of data to be used for testing (default is 0.1)
        seed: Seed for reproducibility (default is None)

    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test
    """
    assert np.isclose(training_ratio + validation_ratio + test_ratio, 1.0), "Ratios must sum to 1."

    num_samples = X.shape[0]
    indices = np.arange(num_samples)

    if seed is not None:
        np.random.seed(seed)

    np.random.shuffle(indices)

    # Compute the split indices
    training_end = int(training_ratio * num_samples)
    validation_end = int((training_ratio + validation_ratio) * num_samples)

    # Split the data
    train_indices = indices[:training_end]
    val_indices = indices[training_end:validation_end]
    test_indices = indices[validation_end:]

    X_train, Y_train = X[train_indices], Y[train_indices]
    X_val, Y_val = X[val_indices], Y[val_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def plot_loss(train_loss_history, val_loss_history):

    plt.plot(np.arange(len(train_loss_history)), train_loss_history,
             label="Training loss")
    plt.legend()
    plt.xlabel("Iterations (number of batches)")
    plt.ylabel("Loss")

    plt.plot(np.arange(len(val_loss_history)),
             val_loss_history, label="Validation loss")
    plt.legend()
    plt.xlabel("Iterations (number of batches)")
    plt.ylabel("Loss")

    plt.show()
