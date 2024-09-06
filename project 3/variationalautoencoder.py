# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:12:15 2024

@author: Mattia
"""
import torch 
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
def elbo_loss_function(recon_x, x, mu, logvar):
    
    BCE = binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = BCE + KLD
    
    return loss

class VAE(nn.Module):
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
            nn.BatchNorm1d(256)
            
            # return Tensor(batch, self.latent_size)
        )
        
        self.mu = nn.Linear(256, self.latent_size)
        self.log_var = nn.Linear(256, self.latent_size)
         
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
            
            z_mean, z_log_var = self.encode(x[:,0:1,:,:])
            z = self._sampling(z_mean, z_log_var)
            decoded1 = self.decode(z)
            
            z_mean, z_log_var = self.encode(x[:,1:2,:,:])
            z = self._sampling(z_mean, z_log_var)
            decoded2 = self.decode(z)
            
            z_mean, z_log_var = self.encode(x[:,2:3,:,:])
            z = self._sampling(z_mean, z_log_var)
            decoded3 = self.decode(z)
            
            decoded = torch.cat((decoded1, decoded2, decoded3), dim=1)
            
            return decoded, 0, 0
        
        else:
        
            z_mean, z_log_var = self.encode(x)
            
            z = self._sampling(z_mean, z_log_var)
                    
            decoded = self.decode(z)
            
            return decoded, z_mean, z_log_var
    
    def encode(self, x):
        out = self.encoder(x)
        
        return self.mu(out), self.log_var(out)
    
    def decode(self, z):
        x_hat = self.decoder(z)
    
        #output = torch.sigmoid(x_hat)
        return x_hat #output
    
    def _sampling(self, z_mean, z_log_var):
        
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch, dim)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon
    
        '''
        standard_deviation = torch.exp(0.5 * z_log_var)
        epsilon = torch.randn_like(standard_deviation)
        return epsilon * standard_deviation + z_mean
    '''