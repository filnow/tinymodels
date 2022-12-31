import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from collections import OrderedDict
from utils import class_img


class EncoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()        
        self.norm1 = nn.LayerNorm(96)
        self.attn = nn.MultiheadAttention()
        self.norm2 = nn.LayerNorm(96)
        self.mlp = nn.Sequential(
            nn.Linear(96,384),
            nn.Dropout(),
            nn.GELU(),
            nn.Linear(384,96)
        )
    
class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        

class SwinTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()


        self.features = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=4),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(96)
            ),

        )
        self.norm = nn.LayerNorm(768)
        self.head = nn.Linear(768,1000)


data = load_state_dict_from_url('https://download.pytorch.org/models/swin_t-704ceda3.pth')

for i in data.keys():
    print(i, data[i].shape)
