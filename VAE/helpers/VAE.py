
# from PIL import Image
# import os
# import glob
# import re
# from tqdm import tqdm
# import pandas as pd

# from torchvision import transforms

# from torch import optim

# from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
# import numpy as np

import torch
from torch import nn
import torch.nn.functional as F 
        
class VariationalAutoencoder(nn.Module):
    
    def __init__(self, input_dim, z_dim=16, ch_start=4):
        """
        input_dim: flattened input size of the image
        z_dim: dimension of the bottleneck (dimesnionality of the bottleneck)
        """
        super().__init__()
        
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.encoder_output_size = int(self.input_dim/8)
        ########### ENCODER ###########
        
        # N, C, H, W = Batch, 1, 255, 255 
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=ch_start, kernel_size=3, stride=2, padding=0), # N, C, H, W = Batch, 16, 128, 128 
            nn.ReLU(),
            nn.Conv2d(ch_start, ch_start*2, kernel_size=3, stride=4, padding=0), # N, C, H, W = Batch, 32, 32, 32 
            nn.ReLU(),
            nn.Flatten() # N, L = = Batch, 32*32*32
        )
        
        self.h_dim = ch_start*2*(self.encoder_output_size)**2 #encoder block output size
        self.fc_2mu = nn.Linear(self.h_dim, self.z_dim)
        self.fc_2sigma = nn.Linear(self.h_dim, self.z_dim)
        
        ########### DECODER ###########
        self.z_2hid = nn.Linear(self.z_dim, self.h_dim)
        
        # N, L
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (ch_start*2, self.encoder_output_size, self.encoder_output_size)), # unflatten L (dim=1) -> C, H, W as in encoder layer
            nn.ConvTranspose2d(in_channels=ch_start*2, out_channels=ch_start, kernel_size=3, stride=4, padding=0),
            nn.Conv2d(ch_start, ch_start, kernel_size=3, padding=1),
            nn.ReLU(),
#             nn.Upsample(scale_factor=(2,2), mode='bilinear'), 

            nn.ConvTranspose2d(in_channels=ch_start, out_channels=3, kernel_size=3, stride=2, padding=0),
            nn.Conv2d(3, 1, kernel_size=3, padding=1)

        )
        
    def reparametrize(self, mu, logvar):
        """
        we predict the mean (mu) and log-variance (logvar) of the latent variables. 
        Using logvar instead of var: This is a common trick for ensuring the variance is always positive 
        (since the exponential of any number is always positive).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def encode(self, x):
        # q_phi(z|x)
        h = self.encoder(x)
        mu, sigma = self.fc_2mu(h), self.fc_2sigma(h)
        return mu, sigma
        
    def decode(self, x):
        # p_theta(x|z)
        h = self.z_2hid(x)
        decoded = torch.sigmoid(self.decoder(h)) #the resulting pixel value needs to be between 0 and 1 
        resized_decoded = F.interpolate(decoded, size=(self.input_dim, self.input_dim), mode='bilinear')
        return resized_decoded 
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        z_reparametrized = self.reparametrize(mu, sigma)
        x_reconstructed = self.decode(z_reparametrized)
        
        return x_reconstructed, mu, sigma, z_reparametrized



class SupervisedVariationalAutoencoder(nn.Module):
    
    def __init__(self, input_dim, z_dim=16, ch_start=4):
        """
        input_dim: flattened input size of the image
        z_dim: dimension of the bottleneck (dimesnionality of the bottleneck)
        """
        super().__init__()
        
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.encoder_output_size = int(self.input_dim/8)
        ########### ENCODER ###########
        
        # N, C, H, W = Batch, 1, 255, 255 
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=ch_start, kernel_size=3, stride=2, padding=0), # N, C, H, W = Batch, 16, 128, 128 
            nn.ReLU(),
            nn.Conv2d(ch_start, ch_start*2, kernel_size=3, stride=4, padding=0), # N, C, H, W = Batch, 32, 32, 32 
            nn.ReLU(),
            nn.Flatten() # N, L = = Batch, 32*32*32
        )
        
        self.h_dim = ch_start*2*(self.encoder_output_size)**2 #encoder block output size
        self.fc_2mu = nn.Linear(self.h_dim, self.z_dim)
        self.fc_2sigma = nn.Linear(self.h_dim, self.z_dim)
        
        self.cls = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.z_dim, 1))

        ########### DECODER ###########
        self.z_2hid = nn.Linear(self.z_dim, self.h_dim)
        
        # N, L
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (ch_start*2, self.encoder_output_size, self.encoder_output_size)), # unflatten L (dim=1) -> C, H, W as in encoder layer
            nn.ConvTranspose2d(in_channels=ch_start*2, out_channels=ch_start, kernel_size=3, stride=4, padding=0),
            nn.Conv2d(ch_start, ch_start, kernel_size=3, padding=1),
            nn.ReLU(),
#             nn.Upsample(scale_factor=(2,2), mode='bilinear'), 

            nn.ConvTranspose2d(in_channels=ch_start, out_channels=3, kernel_size=3, stride=2, padding=0),
            nn.Conv2d(3, 1, kernel_size=3, padding=1)

        )
        
    def reparametrize(self, mu, logvar):
        """
        we predict the mean (mu) and log-variance (logvar) of the latent variables. 
        Using logvar instead of var: This is a common trick for ensuring the variance is always positive 
        (since the exponential of any number is always positive).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def encode(self, x):
        # q_phi(z|x)
        h = self.encoder(x)
        mu, sigma = self.fc_2mu(h), self.fc_2sigma(h)
        return mu, sigma
        
    def decode(self, x):
        # p_theta(x|z)
        h = self.z_2hid(x)
        decoded = torch.sigmoid(self.decoder(h)) #the resulting pixel value needs to be between 0 and 1 
        resized_decoded = F.interpolate(decoded, size=(self.input_dim, self.input_dim), mode='bilinear')
        return resized_decoded 
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        z_reparametrized = self.reparametrize(mu, sigma)
        y_pred = self.cls(z_reparametrized)
        x_reconstructed = self.decode(z_reparametrized)
        
        return x_reconstructed, mu, sigma, z_reparametrized, y_pred
 



