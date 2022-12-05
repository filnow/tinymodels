import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.hub import load_state_dict_from_url
import requests
from io import BytesIO
import sys




class Inception(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()


    def forward(self, x):
        pass




model = Inception()
data = load_state_dict_from_url('https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth')
print(data.keys())