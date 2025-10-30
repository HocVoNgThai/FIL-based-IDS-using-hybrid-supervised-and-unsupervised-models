import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class AdvancedDimReducerAE(nn.Module):
    """
    Model 1: Giảm chiều dữ liệu (46D -> 32D). Input -> 128 -> 64 -> 32.
    """
    
    def __init__(self, input_dim, latent_dim=32):
        super(AdvancedDimReducerAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z