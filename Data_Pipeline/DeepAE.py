import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class DeepAnomalyAE(nn.Module):
    """
    Model 2: Phát hiện bất thường (Kiến trúc robust 32D -> 8D).
    Increased layer width (capacity), tight bottleneck, increased dropout.
    """
    def __init__(self, input_dim=32, latent_dim=8): # <<< Bottleneck 8D
        super(DeepAnomalyAE, self).__init__()
        # Input (32) -> 256 -> 128 -> 8 (Latent)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),      # <<< Increased width
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),                # <<< Increased dropout
            nn.Linear(256, 128),      # <<< Increased width
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),                # <<< Increased dropout
            nn.Linear(128, latent_dim) # To 8
        )
        # 8 (Latent) -> 128 -> 256 -> 32 (Output)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),      # <<< Increased width
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),                # <<< Increased dropout
            nn.Linear(128, 256),      # <<< Increased width
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),                # <<< Increased dropout
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed
    
