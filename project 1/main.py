# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 09:43:26 2024

@author: Mattia
"""
from data_generator import ImgGenerator
from data_generator_parser import DataGeneratorConfig
from neural_network_parser import NeuralNetworkConfig
from neural_network import NeuralNetwork 
from utils import split_dataset, accuracy, plot_loss

if __name__ == "__main__":
    data_generator_config = DataGeneratorConfig()
    data_generator_config.get_config('data_generator.ini')
    
    images_generator = ImgGenerator(data_generator_config)
    
    #on/off
    #images_generator.create_images() 
    
    #images_generator.print_samples()
    data, targets = images_generator.get_images(flat=True)
    
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(data, targets)
        
    nn_param = NeuralNetworkConfig()
    #nn_param.get_config('neural_network_config1.yml')
    #nn_param.get_config('neural_network_config2.yml')
    nn_param.get_config('neural_network_config3.yml')
    
    ciccia = nn_param.layers
    
    neural_network = NeuralNetwork(nn_param.learning_rate,
                                   nn_param.layers, nn_param.softmax,
                                   nn_param.loss_function, nn_param.regularaizer, 
                                   nn_param.regularaizer_factor, 
                                   nn_param.initial_weight_ranges, 
                                   nn_param.initial_bias_ranges, nn_param.verbose)
    
    res = neural_network.get_weights()
    
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(
        data, targets)
    
    train_loss_history, val_loss_history = neural_network.train(
        nn_param.epochs, nn_param.batch_size, X_train, Y_train, X_val, Y_val)

    test_output = neural_network.predict(X_test, False)
    print(test_output)
    test_accuracy = accuracy(test_output.T, Y_test.T)
    
    print("test accuracy: ", test_accuracy)
    
    plot_loss(train_loss_history, val_loss_history)
