# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 09:46:46 2024

@author: Mattia
"""
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, num_features, hidden_units, output_size, num_layers=1, dropout_rate=0):
        '''
        Parameters
        ----------
        num_features : int
            number of feature/dimension of each sample.
        hidden_units : int
            how many neurons for each neural net.
        output_size : int
            number of outputs values
        num_layers : int, optional
            Number of stacked lstm. The default is 1.
        dropout_rate : float, optional
            The default is 0.
        '''
        super().__init__()
        self.num_features = num_features
        
        #lstm has 3 standard gates with 4 neural net inside each cell (don t know pytorch)
        # Defining the number of layers and the nodes in each layer
        self.hidden_units = hidden_units    #how many neurons for each neural net              
        self.num_layers = num_layers    #number of lstm stacked    
        self.output_size = output_size
        self.dropout = dropout_rate

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout_rate
        )
        # Fully connected layer
        self.fc1 = nn.Linear(in_features=self.hidden_units, out_features=self.hidden_units)
        self.LeakyReLU = nn.LeakyReLU()
        self.fc2 = nn.Linear(self.hidden_units, out_features=self.output_size)


    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.fc1(hn[0]) #.flatten()  # First dim of Hn is num_layers, which is set to 1 above
        out = self.LeakyReLU(out)
        out = self.fc2(out)
        return out.squeeze()
    
class CNN(nn.Module):
    def __init__(self, input_size, output_size, in_channels, out_channels, kernel_size):
        '''
        Parameters
        ----------
        input_size : int
            input lenght: how many samples
        output_size : int
            number of outputs values
        in_channels : int
            n_feature/dimensions of the input (how many dimension for each sample)
        out_channels : int
            defined by the number of kernel/filter. how many information want to
            select/capture.
        kernel_size : int
            dimension of each kernel.
        '''
        
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size,  padding=1)
        
        self.relu = nn.ReLU(inplace=True) #inplace=True means modify direcly without allocating new memory
        self.tanh = nn.Tanh()
        self.LeakyReLU = nn.LeakyReLU()
        s = input_size*out_channels
        self.fc1 = nn.Linear(s, s//2)
        self.fc2 = nn.Linear(s//2, s//4)
        self.fc3 = nn.Linear(s//4, output_size)
        
    def forward(self,x):
        x = self.conv1d(x) #(Batch,H,L)
        x=x.reshape(x.size(0), -1) #(Batch,H*L)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.LeakyReLU(x)
        x = self.fc3(x)
        
        '''
        x = self.relu(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        '''
        
        return x.squeeze()
    
class FeedForward(nn.Module):
    def __init__(self, input_size, output_size):
        
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_size, input_size)
        self.LeakyReLU = nn.LeakyReLU()
        self.fc5 = nn.Linear(input_size, input_size//2)
        
        self.fc3 = nn.Linear(input_size//2, input_size//4)
        self.tanh = nn.Tanh()
        self.fc4 = nn.Linear(input_size//4, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        #x = self.tanh(x)
        #x = self.fc2(x)
        #x = self.LeakyReLU(x)
        x = self.tanh(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        
        return x.squeeze()