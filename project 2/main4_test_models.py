# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:41:15 2024

@author: Mattia
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import split_sequence, train_validate_test_split, CustomDataset, plot_bar
from models import LSTM, CNN, FeedForward as FF
from feature_eng import feature_engenering

from train_eval_test_model import train_model, eval_model, test_model, plot_forecast, plot_loss


if __name__ == "__main__":
    path_training_dataset = "consumption_and_temperatures.csv"
    path_hold_out_test_set = "test_set.csv"
    #path_hold_out_test_set = "consumption_and_temperatures.csv"
    
    scaler = StandardScaler() #[None]
    loss_function = nn.MSELoss()   

    models = ['CNN','FF','LSTM']
    #models = ['FF']
    #regions = ['NO1','NO2','NO3','NO4','NO5']
    regions = ['NO3']
    
    model_name_list = []
    mae_list = []
    
    input_lenght=24
    output_lenght=24
    num_test_plot=10
    is_plot_bar = 1
    #is_plot_bar = 0
    
    for model in models:
        for region in regions:
            no_labels = ["month", "hour", f"{region}_temperature"] 
            label = ["NO4_consumption"]#[f"{region}_consumption"]
            label = [f"{region}_consumption"]
            features = label+no_labels
            num_features = len(features)
            
            df = pd.read_csv(path_training_dataset)
            df_copy = feature_engenering(df)
            
            raw_data = df_copy[features]
            raw_data = raw_data.to_numpy()
            
            train, validation, test = train_validate_test_split(df_copy)
            
            if path_hold_out_test_set != None:
                df_test = pd.read_csv(path_hold_out_test_set)
                test = feature_engenering(df)
            
            if scaler != None:
                scaler.fit(train[label])
                
                pd.options.mode.chained_assignment = None
                #train[label] = scaler.transform(train[label])
                #validation[label] = scaler.transform(validation[label])
                test[label] = scaler.transform(test[label])
            
            model_name = f'{model}_{region}'
            print(model_name)
            
            ciao = torch.load(f'model_{model_name}.pth')
            
            if model == 'LSTM':
                model_load = LSTM(num_features=num_features, hidden_units=9, output_size=1, num_layers=1, dropout_rate=0)
            elif model == 'CNN':
                model_load = CNN(input_size=input_lenght, output_size=1, in_channels=num_features, out_channels=3, kernel_size=3)
            elif model == 'FF':
                model_load = FF(input_size=num_features*input_lenght, output_size=1)
            
            model_load.load_state_dict(torch.load(f'model_{model_name}.pth'))
            mae = test_model(test, features, label, model_load, model, input_lenght, output_lenght, num_plots=num_test_plot, scaler=scaler)
            print(f"mae: {mae}")
            mae_list.append(mae)
            model_name_list.append(model_name)
       
    if is_plot_bar:
        plot_bar(mae_list, model_name_list, x_name='models', y_name='mae error', title='models compared')
    