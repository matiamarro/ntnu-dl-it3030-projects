# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:22:28 2024

@author: Mattia
"""
import yaml
import numpy as np

class NeuralNetworkConfig:
    def get_config(self, path_name='neural_network.yml'):
        """
        Read the configuration file initialize the configuration object NeuralNetworkConfig

        Parameters
        ----------
        path_name : String, optional
            path of the config file. The default is 'neural_network.yml'.

        Returns
        -------
        None.

        """
        self.yaml_file_path = path_name
        # Read the YAML file
        with open(self.yaml_file_path, 'r') as yaml_file:
            config_data = yaml.safe_load(yaml_file)
        
        # Accessing configuration settings
        self.learning_rate = config_data['learning_rate']
        self.batch_size = config_data['batch_size']
        self.epochs = config_data['epochs']
        self.verbose = config_data['verbose']
        self.layers = config_data['layers']
        self.softmax = config_data['softmax']
        self.loss_function = config_data['loss_function']
        self.regularaizer = config_data['regularaizer']
        self.regularaizer_factor = config_data['regularaizer_factor']
        self.initial_weight_ranges = config_data['initial_weight_ranges']
        self.initial_bias_ranges = config_data['initial_bias_ranges']
    
    def __str__(self):
        """
        Returns
        -------
        str
            the configuration file as string

        """
        feats = vars(self)
        feats_formatted = ', '.join(f"{k}={v}" for k, v in feats.items())
        return f"{self.__class__.__name__}({feats_formatted})"
    
if __name__ == "__main__":
    data_config = NeuralNetworkConfig()
    data_config.get_config()
    print(data_config)
    A = np.array([1,2,3]) * np.array([4, 5, 6])