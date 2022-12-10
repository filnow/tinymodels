import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from utils import class_img




class EfficientNET(nn.Module):
    def __init__(self) -> None:
        super().__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x





model = EfficientNET()

data = load_state_dict_from_url('https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth')

print(data.keys())