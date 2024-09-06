# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 09:51:33 2024

@author: Mattia
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

def train_model(sel_model, data_loader, model, seq_lenght, loss_function, optimizer):
    '''
    Parameters
    ----------
    sel_model : String
        which model is training ['LSTM','FF','CNN']
    data_loader : DataLoader 
    model : torch.nn.Model
    seq_lenght : int
        lenght inout windows.
    loss_function : torch.nn loss function
    optimizer : torch optimizer

    Returns
    -------
    avg_loss, outputs, true_labels

    '''
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    outputs = []
    labels = []
    
    for X, y in data_loader:
        #X[:,-1] = 0.2
        
        if sel_model == 'CNN':
            
            X=X.permute(0, 2, 1) #fitting size required for the model (conv1d)
                                 #(Batch,Lenght,Channels) to (Batch,Channels,Lenght)
            
        elif sel_model == 'FF':
            X=X.view(X.size(0), -1) #fitting size for the model 
                                    #(batch_size, win_lenght, dimension) to (batch_size, win_leng*dimensione)
                                    #flatting the input
        output = model(X)
        
        loss = loss_function(output, y)
        
        outputs.extend(output.tolist())
        labels.extend(y.tolist())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    
    print(f"Train loss: {avg_loss}")
    
    return avg_loss, np.array(outputs), np.array(labels)

def eval_model(sel_model, data_loader, model, seq_lenght, loss_function):
    num_batches = len(data_loader)
    total_loss = 0
    model.eval()
    
    outputs = []
    labels = []
    
    for X, y in data_loader:
        #X[:,-1] = 0.2
        
        if sel_model == 'CNN':
            X=X.permute(0, 2, 1)
            
        elif sel_model == 'FF':
            X=X.view(X.size(0), -1)
            
        output = model(X)
        
        loss = loss_function(output, y)
        
        outputs.extend(output.tolist())
        labels.extend(y.tolist())

        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    
    print(f"++Validation loss: {avg_loss}")
    
    return avg_loss, np.array(outputs), np.array(labels)

def test_model(test_set, features, label, model, sel_model, input_lenght, output_lenght, num_plots, scaler=None):
    test_set_cp = test_set.copy()
    
    count = 0
    
    y_true = []
    y_pred = []
    
    for start in range(0,
                       len(test_set_cp)-output_lenght-input_lenght,
                       input_lenght//2): #shifting the window input
        
        #all input window
        input_window_scaled = (test_set_cp[start:start+input_lenght][features].values)
        #all output window
        output_window_scaled = (test_set_cp[start+input_lenght:start+input_lenght+output_lenght][features].values)
        
        #true_label
        inputoutput_window_scaled = test_set_cp[start:start+output_lenght+input_lenght][label].values
        
        #forecast_label
        test_forecasts_scaled = np.array(predict(input_window_scaled, output_window_scaled, model, sel_model, output_lenght)).reshape(-1, 1) 
        
        print_test_forecasts_scaled = inputoutput_window_scaled.copy()
        print_test_forecasts_scaled[-output_lenght:]=test_forecasts_scaled
        
        if scaler != None:
            inputoutput_window = scaler.inverse_transform(inputoutput_window_scaled)
            print_test_forecasts = scaler.inverse_transform(print_test_forecasts_scaled)
        else:
            inputoutput_window = inputoutput_window_scaled
            print_test_forecasts = print_test_forecasts_scaled
        
        y_true.append(inputoutput_window[-output_lenght:].flatten().tolist())
        y_pred.append(print_test_forecasts[-output_lenght:].flatten().tolist())
        
        if count < num_plots:
            plot_forecast(inputoutput_window, 
                          print_test_forecasts,
                          name=f'{output_lenght} point forecasted - {sel_model}')
            
            plot_error(inputoutput_window, 
                       print_test_forecasts,
                       name=f'absolute error - {sel_model}')
            
            count += 1
    
    mae = mean_absolute_error(y_true, y_pred)
    
    return mae
        
def predict(input_window, output_window, model, sel_model, len_forecast):
    '''
    n in 1 out prediction. from a window of n element try to predict the 1 next step
    then the windows is shifted to predict another time, this steps repetet 
    'len_forecast' times to full fill all the output window

    Parameters
    ----------
    input_window : numpyarray (windows_input_lenght, dimensions)
        input window lenght.
    output_window : numpyarray (windows_output_lenght, dimensions)
        output window lenght..
    model : torch.nn Model
    sel_model : String
    len_forecast : int
        size of the output window.

    Returns
    -------
    forecasts : TYPE
        DESCRIPTION.

    '''
    win_copy = input_window.copy()
    
    forecasts = []
    
    for pred_no in range(len_forecast):
        
        tensor_window = torch.Tensor(np.expand_dims(win_copy, axis=0))
        
        if sel_model == 'CNN':
            tensor_window=tensor_window.permute(0, 2, 1)
        elif sel_model == 'FF':
            tensor_window=tensor_window.view(tensor_window.size(0), -1)
        
        #print(tensor_window)
        forecast = model(tensor_window)
        
        forecast = forecast.item()
        
        forecasts.append(forecast)
        
        next_ = output_window[pred_no] #sample of the feature with all the features needed
        next_[0]=forecast #full filling the forecast did right now
        
        win_copy = np.vstack([win_copy[1:], next_]) #adding the forecast to the input window
            
    return forecasts

def plot_loss(losses, freq=1, name='loss flow'):
    
    epochs = range(0, (len(losses))*freq, freq)

    plt.plot(epochs, losses, linestyle='-')
    
    plt.title(name)
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.show()
    
def plot_error(labels, outputs, name='errors'):
    
    plt.figure(figsize=(8, 6))
    plt.plot(labels-outputs, label='Error', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.title(name)
    plt.legend()
    plt.grid(True)
    
    plt.show()
    
def plot_forecast(labels, outputs, name='forecasts'):
    
    plt.figure(figsize=(8, 6))
    plt.plot(labels, label='True Targets', color='blue')
    plt.plot(outputs, label='Predictions', linestyle='dashed', color='red')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title(name)
    plt.legend()
    plt.grid(True)
    
    plt.show()

'''
if __name__ == "__main__":    
    batch_size=64
    lr=0.001
    #features = ["NO1_consumption", "NO1_temperature"]
    features = ["NO1_consumption"]
    label = ["NO1_consumption"]
    num_features = len(features)
    output_lenght = 24
    input_lenght = 24
    num_epochs=2
    frequence_validation = 1
    frequence_plot_training = 15
    sel_model = "LSTM" #['LSTM','CNN','FF']
    
    #change for n features
    df = pd.read_csv("consumption_and_temperatures.csv")
    df.set_index('timestamp', inplace=True)
    
    df_copy = df.copy()
    raw_data = df_copy[features]
    raw_data = raw_data.to_numpy()
    
    scaler = StandardScaler()
    df_copy[label] = scaler.fit_transform(df_copy[label])
    
    train, validation, test = train_validate_test_split(df_copy)
    
    train_x, train_y = split_sequence(train, features, label, input_lenght)
    val_x, val_y = split_sequence(validation, features, label, input_lenght)
    
    train_dataset = CustomDataset(train_x, train_y)
    val_dataset = CustomDataset(val_x, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    if sel_model == 'LSTM':
        model = LSTM(num_features=num_features, hidden_units=3, output_size=1, num_layers=1, dropout_rate=0)
    elif sel_model == 'CNN':
        model = CNN(input_size=input_lenght, output_size=1, in_channels=1, out_channels=3, kernel_size=3)
    elif sel_model == 'FF':
        model = FF(input_size=num_features*input_lenght, output_size=1)
    
    loss_function = nn.MSELoss()
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
    
    
    test_model(test, features, label, model, sel_model, input_lenght, output_lenght, num_plots=4)
'''   
    
    
