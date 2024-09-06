# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 09:48:34 2024

@author: Mattia
"""
import numpy as np
import torch
import math
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def split_sequence(sequence, features, label, n_steps):
    '''
    Parameters
    ----------
    sequence : Dataframe of shape (num_samples, num_features)
    features : list of string
        features of the dataset
    label : list of string
        labels of the dataset
    n_steps : int
        windows lenght

    Returns
    -------
    numpy_array of shape (num_windows, window_lenght, dimensions_single_sample)
    numpy_array of shape (num_windows,)

    '''
    
    x, y = [], []
    labels = sequence[label].values
    raw_data = sequence[features].values
    
    for i in range(len(sequence)):
        
        end_ix = i + n_steps
        
        if end_ix > len(sequence)-1:
            break
        
        seq_x = raw_data[i:end_ix,:]
        seq_y = labels[end_ix]
        
        x.append(seq_x)
        y.append(seq_y)
        
    #return np.array(x)[:,:,None], np.array(y)
    return np.array(x), np.array(y).squeeze()

def train_validate_test_split(timeseries, train_size=0.8, val_size=0.1, test_size=0.1):
    num_samples = int(len(timeseries) * train_size) # use a parameter to control training size
    train_data = timeseries[:num_samples]
    
    num_samples1 = math.floor((num_samples+len(timeseries)*val_size))
    
    val_data = timeseries[num_samples:num_samples1]
    test_data = timeseries[num_samples1:]
    
    return train_data, val_data, test_data

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.Tensor(data).float()
        self.targets = torch.Tensor(targets).float()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        target = self.targets[index]
        sample = self.data[index]
        
        return sample, target
    
def plot_bar(values, labels, x_name='x', y_name='values', title='bar plot'):
    '''
    #example
    maes = [500, 400, 400]
    labels = ['lstm','cnn','ff']
    
    plot_bar(maes, labels, x_name='models', y_name='mae error', title='models compared')
    '''
    
    
    # Check if the lengths of values and labels are the same
    if len(values) != len(labels):
        raise ValueError("The lengths of values and labels must be the same.")
    
    # Create a bar plot
    plt.figure(figsize=(8, 6))  # Optional: Adjust the figure size
    plt.bar(labels, values)
    
    # Add labels and title
    plt.xlabel(x_name)
    plt.xticks(rotation=90)
    plt.ylabel(y_name)
    plt.title(title)
    
    # Show plot
    plt.show()
