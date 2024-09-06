# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:13:17 2024

@author: Mattia
"""
import torch
from torch.utils.data import DataLoader
from main import get_data_mode, get_dataset, create_model,get_loss_function, generate_imgs
from main import plot_images, anomaly_VAE_detection, train_VAE_model, model_validation
from verification_net import VerificationNet 
    
if __name__ == '__main__':    
    set_configuration = 'color_binary_complete'
    model_name = 'vae'
    latent_size =  70#to-do
    lr = 0.0008
    epochs = 30
    batch_size = 64
    load_model = True
    model_save = False
    model_save_name = f'{model_name}_{set_configuration}_{latent_size}latent_6.pth'
    model_load_name = 'models/vae_mono_binary_missing_70latent_4_server.pth'
    #model_load_name = 'ae_mono_binary_complete_256latent_1.pth'
    
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
            
            if model_name == 'vae':
                #model.load_state_dict(torch.load(model_load_name))#cancel
                
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
        
    
        #   Generation VAE
        num_samples = 10
        generated_imgs = generate_imgs(num_samples, latent_size, model, color=True)
        
        plot_images(true_img=generated_imgs, recostructed_img=None, labels=None)
        
        
        '''
        #   Anomaly Detector VAE
        top_anomaly = 20
        
        anomaly_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        indexes, label_vec, loss_vec = anomaly_VAE_detection(model, anomaly_dataloader)
        top_indexes = indexes[:top_anomaly]
        top_labels = label_vec[:top_anomaly]
        print(f"top label detected: {top_labels}")
        plot_images(true_img=dataset.data[top_indexes], recostructed_img=None, labels=top_labels.tolist())
        '''
        
        '''
        #   plot some VAE mono comparison recostructed
        num_plot = 30
        val_dataloader = DataLoader(dataset, batch_size=num_plot, shuffle=True)
        data, label = next(iter(val_dataloader))
        output,_,_ = model(data)
        plot_images(true_img=data, recostructed_img=output, labels=label)
        '''
        
        '''
        #   Validation 
        model_validation(model, dataset, model_name, color=True)
        '''
        
        
        