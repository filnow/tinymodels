import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from collections import OrderedDict
from utils import class_img


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relative_position_bias_table = nn.Parameter(torch.zeros(169,3))
        self.relative_position_index = nn.Parameter(torch.zeros(2401))
        
        self.qkv = nn.Conv2d(96, 288, kernel_size=3)
        self.proj = nn.Conv2d(96, 288, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x + self.relative_position_bias_table
        x = x + self.relative_position_index

        x = self.qkv(x)
        x = self.proj(x)

        return x
    

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()        
        self.norm1 = nn.LayerNorm(96)
        
        self.attn = Attention()
        
        self.norm2 = nn.LayerNorm(96)
        
        self.mlp = nn.Sequential(
            nn.Linear(96,384),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(384,96)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x  = self.attn(self.norm1(x)) + x
        
        return self.mlp(self.norm2(x)) + x
        
 
class SwinTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self.conv(),
            self.encoder(),
            #self.reduction(384,192),
        )
        
        self.norm = nn.LayerNorm(768)
        
        self.head = nn.Linear(768,1000)

    @staticmethod
    def reduction(inch: int, outch: int):
        return OrderedDict(
                [
                    ("reduction", nn.Conv2d(inch, outch, kernel_size=1, bias=False)),
                    ("norm", nn.BatchNorm2d(inch)),
                ])

    @staticmethod
    def conv():
        return nn.Sequential(
                    nn.Conv2d(3, 96, kernel_size=4), 
                    nn.ReLU(inplace=True), 
                    nn.LayerNorm(96)
        )
    
    @staticmethod
    def encoder(): return nn.Sequential(Encoder(), Encoder())
        



data = load_state_dict_from_url('https://download.pytorch.org/models/swin_t-704ceda3.pth')

model = SwinTransformer()
model.load_state_dict(data)
model.eval()

for i in data.keys():
    print(i, data[i].shape)
