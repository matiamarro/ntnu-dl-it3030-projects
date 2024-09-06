# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:13:06 2024

@author: Mattia
"""
import torch 
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class AE(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size

        self.encoder = torch.nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.05),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.05),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.05),
            nn.BatchNorm2d(256),
            nn.Flatten(),
            nn.Linear(256*4*4, 256),
            
            nn.LeakyReLU(0.05),
            nn.BatchNorm1d(256),
            nn.Linear(256, self.latent_size)
            
            
            # return Tensor(batch, self.latent_size)
        )
         
        self.decoder = torch.nn.Sequential(
            nn.Linear(self.latent_size, 256),
            nn.LeakyReLU(0.05),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256*4*4),
            nn.LeakyReLU(0.05),
            nn.BatchNorm1d(256*4*4),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.05),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.05),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        if x.size(1) == 3:
            #prova = x[:,0:1,:,:]
            encoded1 = self.encode(x[:,0:1,:,:])
            decoded1 = self.decode(encoded1)
            encoded2 = self.encode(x[:,1:2,:,:])
            decoded2 = self.decode(encoded2)
            encoded3 = self.encode(x[:,2:3,:,:])
            decoded3 = self.decode(encoded3)
            
            decoded = torch.cat((decoded1, decoded2, decoded3), dim=1)
        else:
            encoded = self.encode(x)
            decoded = self.decode(encoded)
        
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

        