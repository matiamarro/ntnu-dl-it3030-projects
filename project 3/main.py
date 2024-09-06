# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:39:42 2024

@author: Mattia
"""
from stacked_mnist import DataMode, StackedMNISTData
from autoencoder import AE
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from verification_net import VerificationNet  
import numpy as np
from variationalautoencoder import VAE, elbo_loss_function
import torch.nn.functional as F

def get_data_mode(set_configuration):
    data_modes = {
        'mono_float_complete': DataMode.MONO,
        'mono_float_missing': DataMode.MONO | DataMode.MISSING,
        'mono_binary_complete': DataMode.MONO | DataMode.BINARY, 
        'mono_binary_missing': DataMode.MONO | DataMode.BINARY | DataMode.MISSING,
        'color_float_complete': DataMode.COLOR,
        'color_float_missing': DataMode.COLOR | DataMode.MISSING,
        'color_binary_complete': DataMode.COLOR | DataMode.BINARY,
        'color_binary_missing': DataMode.COLOR | DataMode.BINARY | DataMode.MISSING
    }

    return data_modes[set_configuration]

def get_loss_function(set_configuration):
    '''
    BCE is Binary Cross Entropy

    '''
    loss_functions = {
        'mono_float_complete': torch.nn.MSELoss(),
        'mono_float_missing': torch.nn.MSELoss(),
        'mono_binary_complete': torch.nn.BCELoss(), 
        'mono_binary_missing': torch.nn.BCELoss(),
        'color_float_complete': torch.nn.MSELoss(),
        'color_float_missing': torch.nn.MSELoss(),
        'color_binary_complete': torch.nn.BCELoss(),
        'color_binary_missing': torch.nn.BCELoss()
    }

    return loss_functions[set_configuration]

def get_dataset(data_mode, set_configuration):
    '''
    
    Parameters
    ----------
    data_mode : es. 'DataMode.MONO | DataMode.MISSING'
    set_configuration : es. 'mono_float_missing'

    Returns
    -------
    dataset : StackedMNISTData 
        StackedMNISTData it's a subclass of torch.utils.data.Dataset that
        contains 'data' and 'targets' tensors.
            - data is Tensor(batch, channels, img_dim1, img_dim2)
            - target is Tensor(batch,)
    '''
    
    dataset = StackedMNISTData(mode=data_mode)
    
    if 'color' in set_configuration:
        dataset.data = dataset.data.permute(0,3,1,2)
    else:
        dataset.data = dataset.data.unsqueeze(1) 
    
    return dataset
      
def create_model(model_name, latent_size, binary=False):
    if model_name == 'ae':
        ae = AE(latent_size)
        return ae
    
    if model_name == 'vae':
        ae = VAE(latent_size)
        return ae

def train_AE_model(model, dataloader, loss_function, optimizer):
    '''
    Parameters
    ----------
    model :  Torch AE or VAE model from class autoencoder.py
    dataloader : DataLoader
    loss_function : torch.nn loss object
        which loss function use to train the model
    optimizer : torch.optim 
        which torch optimizer use to train the model

    Returns
    -------
    losses : TYPE
        DESCRIPTION.
    '''
    #todo: managing 3 channels
    
    i = 0 
    
    losses = []
    
    model.train()
    for batch in dataloader:
        
        X = batch[0]
        
        #X = X.view(X.size(0), 1, 28, 28) 
        y = batch[1]
        
        output = model(X)
        
        loss = loss_function(output, X)
        #print(loss.item())
        
        loss.backward()
        
        optimizer.step()
        
        losses.append(loss.item())
        
        i += 1
        
        if i % 180 == 0:
            plot_images(X[0], output[0], y)
            
    return losses

def train_VAE_model(model, dataloader, optimizer):
    '''
    Parameters
    ----------
    model :  Torch AE or VAE model from class autoencoder.py
    dataloader : DataLoader
    loss_function : torch.nn loss object
        which loss function use to train the model
    optimizer : torch.optim 
        which torch optimizer use to train the model

    Returns
    -------
    losses : TYPE
        DESCRIPTION.
    '''
    #todo: managing 3 channels
    loss_function = elbo_loss_function
    
    i = 0 
    
    losses = []
    
    model.train()
    for batch in dataloader:
        
        X = batch[0]
        
        #X = X.view(X.size(0), 1, 28, 28) 
        y = batch[1]
        
        output, z_mean, z_log_var = model(X)
        
        is_between_zero_and_one = (output >= 0) & (output <= 1)
        all_between_zero_and_one = is_between_zero_and_one.all()
        
        if all_between_zero_and_one:
            loss = loss_function(output, X, z_mean, z_log_var)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            i += 1
            
        else: 
            print("output is not between 0 and 1")
            print(output.detach().numpy())
            
        
        if i % 300 == 0:
            plot_images(X[0], output[0], y)
            
    return losses
            
        
def plot_images(true_img=None, recostructed_img=None, labels=None):
    '''
    true_img and recostructed_img: Tensor(batch,channel,dim1,dim2)
    label: Tensor(batch,)
    
    we can have diff cases:
        - want to compare true and autoencoded imgs
        - want to plot creates images. recostructed_img=None, labels=None
        
    '''
    if true_img!=None and recostructed_img != None and labels!=None:
        
        for i in range(true_img.size(0)):
        
            img1 = true_img[i].squeeze().detach().numpy().reshape(28,28)
            img2 = recostructed_img[i].squeeze().detach().numpy().reshape(28,28)
            label = labels[i]
            
            # Plotting
            plt.figure(figsize=(10, 5))  # Set figure size (width, height)
            plt.suptitle(f'Number: {label}', fontsize=16)  # Set main title of the entire plot
            
            # Plot image 1 (Input Image)
            plt.subplot(1, 2, 1)  # (rows, columns, panel number)
            plt.imshow(img1, cmap='gray')  # Assuming grayscale images
            plt.title('Input Image')  # Set title for the left subplot
            plt.axis('off')  # Disable axis
            
            # Plot image 2 (Output Image)
            plt.subplot(1, 2, 2)
            plt.imshow(img2, cmap='gray')
            plt.title('Output Image')  # Set title for the right subplot
            plt.axis('off')
            
            plt.show()
            
    elif recostructed_img == None:
        
        for i in range(true_img.size(0)):
            if true_img.size(1) == 1:
                img1 = true_img[i].squeeze().detach().numpy().reshape(28,28)
                
                
                plt.figure(figsize=(10, 5))
                if labels != None:
                    plt.title(f"label {labels[i]}")
                plt.imshow(img1, cmap='gray') #'gray'
                plt.title(f'Generated Image {i}')
                
                plt.show()
                
            else:
                img = true_img[i].detach().numpy()
                background = np.zeros((28, 28))
                
                img1 =  np.round(img[0])
                img2 =  np.round(img[1])
                img3 =  np.round(img[2])
                
                #background = np.logical_or(np.logical_or(img1, img3), img3)
                
                plt.figure(figsize=(10, 5))
                
                if labels != None:
                    plt.title(f"label {labels[i]}")
                
                plt.imshow(img1, cmap='Reds', alpha=0.5) #'gray'
                plt.imshow(img2, cmap='Greens', alpha=0.5) #'gray'
                plt.imshow(img3, cmap='Blues', alpha=0.5) #'gray'
                #plt.imshow(background, cmap='gray', alpha=0.5)
                
                               
                     
def generate_imgs(num_samples, latent_size, model, color = False):
    '''
    Parameters
    ----------
    num_samples : int
    latent_size : int
    model : Torch AE or VAE model from class autoencoder.py

    Return: Tensor of images (num_samples, num_channels, img_dim1, img_dim2)
    '''
    if color == True:
        outputs = []
        for i in range(3):
           z = np.random.randn(num_samples, latent_size).astype(np.float32)
           
           z_tensor = torch.tensor(z)
           
           output = model.decode(z_tensor)
           
           outputs.append(output)
        
        merged_tensor = torch.cat((outputs[0], outputs[1], outputs[2]), dim=1)
        
        return merged_tensor
        
        
    else:
    
        z = np.random.randn(num_samples, latent_size).astype(np.float32)
        
        z_tensor = torch.tensor(z)
        
        output = model.decode(z_tensor)

        return output
    

def anomaly_AE_detection(model, dataloader, loss_function, dim_dataset=599):
    
    index_anolamy = []
    losses = []
    labels = []
    
    model.eval()
    
    for index, batch in enumerate(dataloader):
        
        X = batch[0]
        y = batch[1].item()
        #X = X.view(X.size(0), 1, 28, 28) 
        
        output = model(X)
        
        loss = loss_function(output, X).item()
        losses.append(loss)
        
        labels.append(y)
            
        if index==dim_dataset:
            break
    
    losses = np.array(losses)
    labels = np.array(labels)
    
    index_order = np.argsort(-np.array(losses))

    loss_ordered = losses[index_order]
    labels_ordered = labels[index_order]
    
    return index_order, labels_ordered, loss_ordered
    
def anomaly_VAE_detection(model, dataloader, dim_dataset=599):
    '''
    Parameters
    ----------
    model : Torch AE or VAE model from class autoencoder.py
    sample : Tensor(batch, channel, img_dim1, img_dim2)

    Returns
    -------
    - array of the lossses
    - np.array of int index of the anomaly images
    es. [4,55,100] are index of the 'samples' array where an anomaly is detected

    '''
    num_samples = 1000
    latent_size = model.latent_size
    rand_latent_vec = torch.randn(num_samples, latent_size)
    
    sample_predictions = model.decode(rand_latent_vec)
    
    p_xz_vec = []
    list_label = []
    
    model.eval()
    
    for index, batch in enumerate(dataloader):
        
        if index == dim_dataset: 
            break
        
        X = batch[0]
        X = X.view(-1, 28, 28)
        y = batch[1].item()
        list_label.append(y)
        #X = X.view(X.size(0), 1, 28, 28) 
        
        loss_vec = []
        loss_vec_1 = []
        loss_vec_2 = []
        loss_vecs = [loss_vec, loss_vec_1, loss_vec_2]
        
        for output in sample_predictions:
            #loss = F.binary_cross_entropy_with_logits(output, X).item()
            #loss = cross_entropy(output.detach().numpy().flatten(), X.detach().numpy().flatten())
            if X.size(0) == 1:
                loss = torch.nn.functional.binary_cross_entropy(output, X).item()
                
                #print(f"{loss}, {loss2}")
                loss = 10 ** loss
                loss_vec.append(loss)
            else:
                for i in range(3):
                    loss = torch.nn.functional.binary_cross_entropy(output, X[i:i+1,:,:]).item()
                    loss = 10 ** loss
                    
                    loss_vecs[i].append(loss)
        if X.size(0) == 1: 
            p_xz = np.mean(loss_vec) 
            #print(f"{p_xz}: number_{y}")
            p_xz_vec.append(p_xz)
        else:
            p_xz_mul = 1
            for i in range(3):
                p_xz = np.mean(loss_vecs[i]) 
                #print(f"{p_xz}: number_{y}")
                p_xz_mul += p_xz
            
            p_xz_vec.append(p_xz_mul)
    
    p_xz_vec = np.array(p_xz_vec)
    list_label = np.array(list_label)
    
    index_order = np.argsort(-p_xz_vec)

    # Riordinare entrambi i vettori utilizzando gli stessi indici
    loss_order = p_xz_vec[index_order]
    label_order = list_label[index_order]
    
    return index_order,label_order, loss_order

def cross_entropy(y_pred, y_true):
    #y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip values to prevent log(0)
    #return -np.sum(y_true * np.log(y_pred+epsilon))
    
    #y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    N = y_pred.shape[0]
    #ce = - np.sum(y_true * np.log(y_pred + 1e-9) + np.sum((1-y_true) * np.log(1-y_pred + 1e-9))) / N
    ce = - np.sum(y_true * np.log(y_pred + 1e-9)+ (1-y_true) * np.log(1-y_pred+1e-9)) / N
    return ce

def model_validation(model, dataset, model_name, color=False):
    tolerance = 0.8
    #   Verification AE
    ver_net = VerificationNet()
    if color== True:
        #ver_net = VerificationNet(file_name="./models/verification_color_mono_binary_complete.weights.h5")
        tolerance = 0.5
    else:
        ver_net = VerificationNet()
    
    test_dataloader = DataLoader(dataset, batch_size=1000, shuffle=True) #1000 samples in the test set
    
    data, label = next(iter(test_dataloader))
    
    #label = dataset.targets
    if model_name == 'vae':
        output,_,_ = model(data)
    else:
        output = model(data)
    
    pred, acc = ver_net.check_predictability(data=data.permute(0,2,3,1).detach().numpy(), 
                                             correct_labels=label.numpy(),
                                             tolerance=tolerance)
    
    #print(f'cov: {cov}')
    print(f'pred: {pred}')
    print(f'acc: {acc}')
    
    return pred, acc
 
if __name__ == '__main__':    
    set_configuration = 'color_binary_complete'
    model_name = 'ae'
    latent_size =  70#to-do
    lr = 0.001
    epochs = 30
    batch_size = 64
    load_model = True
    model_save = True
    model_save_name = f'{model_name}_{set_configuration}_{latent_size}latent_1_2.pth'
    #model_load_name = 'models/vae_mono_binary_missing_70latent_1_server.pth'
    model_load_name = 'models/ae_mono_binary_missing_70latent_1_2.pth'
    
    data_mode = get_data_mode(set_configuration)
    dataset = get_dataset(data_mode, set_configuration) #Dataset
    #img1 = dataset[0][0].numpy()
    #print(img1)
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #plot_images(true_img=dataset.data[:5], recostructed_img=None, labels=None, color = True)
    
    model = create_model(model_name, latent_size)
    
    if load_model == False:
        loss_function = get_loss_function(set_configuration)
        
        
        for i in range(epochs):
            print(f"training {model_name} epoch n: {i+1}")
            if model_name == 'ae':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = 1e-8)
                train_AE_model(model, train_dataloader, loss_function, optimizer)
            elif model_name == 'vae':
                model.load_state_dict(torch.load(model_load_name))#cancel
                
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                train_VAE_model(model, train_dataloader, optimizer)
            
            #test_features, test_labels = next(iter(train_dataloader))
            
            #img_true = test_features[0].view(1,1,28,28) # X = X.view(X.size(0), -1)
            
            #img_recostructed = model(img_true)
            
            #plot_images(img_true, img_recostructed, test_labels)
            
            if model_save == True:
                torch.save(model.state_dict(), model_save_name)
    
    else:
        
        model.load_state_dict(torch.load(model_load_name))
        
        
        #   Anomaly Detector AE
        top_anomaly = 20
        
        anomaly_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        loss_function = get_loss_function(set_configuration)
        
        indexes, labels, losses = anomaly_AE_detection(model, anomaly_dataloader, loss_function, dim_dataset = 600)
        
        top_indexes = indexes[:top_anomaly]
        top_labels = labels[:top_anomaly]
        print(f"top label detected: {labels[:30]}")
        plot_images(true_img=dataset.data[top_indexes], recostructed_img=None, labels=top_labels.tolist())
        
        
        '''
        #   Generation AE mono
        num_samples = 4
        generated_imgs = generate_imgs(num_samples, latent_size, model)
        
        plot_images(true_img=generated_imgs, recostructed_img=None, labels=None)
        '''
        
        '''
        #   Generation AE color
        num_samples = 4
        generated_imgs = generate_imgs(num_samples, latent_size, model, color = True)
        
        plot_images(true_img=generated_imgs, recostructed_img=None, labels=None)
        '''
        
        '''
        #   Validation 
        model_validation(model, dataset, model_name, color=True)
        '''
        
        
        
        #plot_images(data, output, label)
        