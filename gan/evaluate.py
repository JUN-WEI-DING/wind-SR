import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from utils import normalization, common

def inference(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
              gen=None,critic=None,
              z_dim:int=100, N:int=7,
              lrimage=None,
              n_sample:int=10):

    lrimage = torch.as_tensor(lrimage, dtype=torch.float32, device=device)
    
    if lrimage.dim() == 3:
        lrimage = lrimage.unsqueeze(0) 
    
    if lrimage.dim() != 4:
        raise ValueError(f"Expected 4D input (B, C, H, W), but got {lrimage.shape}")

    expanded_lrimage = lrimage.unsqueeze(1).expand(-1, n_sample, *lrimage.shape[1:]).reshape(-1, *lrimage.shape[1:])

    gen=gen.to(device).eval()
    critic=critic.to(device).eval()
    
    with torch.no_grad():
        z = torch.randn(n_sample*lrimage.shape[0], z_dim, *expanded_lrimage.shape[-2:], device=device)
        z[:,:,0::2,1::2]=0
        z[:,:,1::2,0::2]=0
        c = expanded_lrimage
        srimage = gen(z, c)
        low_freq_srimage = common.frequency_filter(srimage, N)
        high_freq_srimage = srimage - low_freq_srimage
        discrim_score = critic(high_freq_srimage, c)
    
    return srimage.cpu(), discrim_score
