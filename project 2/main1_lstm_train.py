# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 09:44:54 2024

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
    
    sel_model = "LSTM" #['LSTM','CNN','FF']
    region = "NO1"
    savemodel = 1
    
    batch_size=64
    lr=0.0006
    #features = ["NO1_consumption", "NO1_temperature"]
    #features = ["NO1_consumption"]
    no_labels = ["month", "hour", f"{region}_temperature"] #, "NO1_temperature"
    label = [f"{region}_consumption"]
    features = label+no_labels
    num_features = len(features)
    output_lenght = 24
    input_lenght = 24
    num_epochs=19
    num_test_plots = 10
    frequence_validation = 1
    frequence_plot_training = 5
    
    scaler = StandardScaler() #[None]
    loss_function = nn.MSELoss()   
    
    df = pd.read_csv("consumption_and_temperatures.csv")
    df_copy = feature_engenering(df)
    
    raw_data = df_copy[features]
    raw_data = raw_data.to_numpy()
    
    train, validation, test = train_validate_test_split(df_copy)
    
    if scaler != None:
        scaler.fit(train[label])
        
        pd.options.mode.chained_assignment = None
        train[label] = scaler.transform(train[label])
        validation[label] = scaler.transform(validation[label])
        test[label] = scaler.transform(test[label])
    
    train_x, train_y = split_sequence(train, features, label, input_lenght)
    val_x, val_y = split_sequence(validation, features, label, input_lenght)
    
    train_dataset = CustomDataset(train_x, train_y)
    val_dataset = CustomDataset(val_x, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    if sel_model == 'LSTM':
        model = LSTM(num_features=num_features, hidden_units=9, output_size=1, num_layers=1, dropout_rate=0)
    elif sel_model == 'CNN':
        model = CNN(input_size=input_lenght, output_size=1, in_channels=1, out_channels=3, kernel_size=3)
    elif sel_model == 'FF':
        model = FF(input_size=num_features*input_lenght, output_size=1)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for ix_epoch in range(1, num_epochs+1):
        print(f"Epoch {ix_epoch}\n---------")
        
        avg_train__loss, outputs, labels = train_model(sel_model, train_loader, model, input_lenght, loss_function, optimizer=optimizer)
        train_losses.append(avg_train__loss)
        
        if ix_epoch % frequence_validation == 0:
            avg_val_loss,_,_ = eval_model(sel_model, val_loader, model, input_lenght, loss_function)
            val_losses.append(avg_val_loss)
        
        if ix_epoch % frequence_plot_training == 0:
            plot_forecast(labels, outputs, name=f'train forecasts {sel_model}')
        
    plot_loss(train_losses, name=f'train loss {sel_model}')
    plot_loss(val_losses, freq=frequence_validation, 
              name=f'validation loss {sel_model}')
    
    mae = test_model(test, features, label, model, sel_model, input_lenght, output_lenght, num_plots=num_test_plots, scaler=scaler)
    print(f"mae: {mae}")
    
    if savemodel:
        print("model saved")
        torch.save(model.state_dict(), f'model_{sel_model}_{region}.pth')