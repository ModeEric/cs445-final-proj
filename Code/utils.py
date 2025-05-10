import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import torch
import math
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  
import numpy as np
import gc
from torchvision.transforms import ToPILImage, ToTensor
import cv2
import torch.nn.functional as F


def generate_heatmap(targets, H=360, W=640, sigma2=10, device='cuda'):
    """
    targets: (batch_size, 3) -> (visibility, x, y)
    H, W: heatmap
    sigma: 
    """
    batch_size = targets.shape[0]
    heatmaps = torch.zeros((batch_size, 1, H, W), device=targets.device)
    for i in range(batch_size):
        visibility, x, y = targets[i]
        if visibility != 0:
            yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
            xx = xx.float()
            yy = yy.float()
            center_x = x.clone().detach()
            center_y = y.clone().detach()
            heatmap = torch.exp(-((xx - center_x)**2 + (yy - center_y)**2) / (2 * sigma2))
            heatmap = heatmap / heatmap.max()  # normalize
            heatmap = heatmap.clone().detach()
            heatmaps[i, 0] = heatmap

    label = (heatmaps*255).long()
    return label



def show_frame(frame, title=''):
    frame = frame.permute(1, 2, 0).numpy()  # (C,H,W) → (H,W,C)
    plt.imshow(frame)
    plt.title(title)
    plt.axis('off')
    plt.show()