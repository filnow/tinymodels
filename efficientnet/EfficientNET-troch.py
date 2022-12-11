import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from utils import class_img

class MBConv(nn.Module):
    def __init__(self, inch, outch, k, s, block0=True) -> None:
        super().__init__()
        
        if block0:
            self._expand_conv = nn.Conv2d(inch, outch, kernel_size=k, stride=s, bias=False)
            self._bn0 = nn.BatchNorm2d(outch)
        
        self._depthwise_conv = nn.Conv2d(1, 120, kernel_size=k, stride=s,  groups=1, bias=False)
        self._bn1 = nn.BatchNorm2d(outch)
        self._se_reduce = nn.Conv2d(inch, outch, kernel_size=k, stride=s)
        self._se_expand = nn.Conv2d(inch, outch, kernel_size=k, stride=s)
        self._project_conv = nn.Conv2d(inch, outch, kernel_size=k, stride=s, bias=False)
        self._bn2 = nn.BatchNorm2d(outch)                   
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        
        ident = x
        
        x = F.silu(self._bn0(self._expand_conv(x)), inplace=True)
        x = F.silu(self._bn1(self._depthwise_conv(x)), inplace=True)
        x = F.silu(self._se_reduce(x), inplace=True)
        x = F.silu(self._se_expand(x), inplace=True)
        x = F.silu(self._bn2(self._project_conv(x)), inplace=True)

        x += ident
        return x

class EfficientNET(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self._conv_stem = nn.Conv2d(3, 32, kernel_size=3, stride=3, bias=False)
        self._bn0 = nn.BatchNorm2d(32)

        self._blocks = nn.Sequential(
            
            MBConv(32,32,1,1, block0=False),
            MBConv(32,32,1,1),
            MBConv(32,32,1,1),
            MBConv(32,32,1,1),
            MBConv(32,32,1,1),
            MBConv(32,32,1,1),
            MBConv(32,32,1,1),
            MBConv(32,32,1,1),
            MBConv(32,32,1,1),
            MBConv(32,32,1,1),
            MBConv(32,32,1,1),
            MBConv(32,32,1,1),
            MBConv(32,32,1,1),
            MBConv(32,32,1,1),
            MBConv(32,32,1,1),
            MBConv(32,32,1,1)

        )

        self._conv_head = nn.Conv2d(64, 64, kernel_size=3, stride=3, bias=False)
        self._bn1 = nn.BatchNorm2d(64)
        self._fc = nn.Linear(64, 1000)
        self.dropout = nn.Dropout()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = F.silu(self._bn0(self._conv_stem(x)), inplace=True)
        x = self._blocks(x)
        x = F.silu(self._bn1(self._conv_head(x)), inplace=True)
        x = self.avgpool(x)
        x = torch.flatten(x)
        x = self._fc(self.dropout(x))

        return x


model = EfficientNET()

#data = load_state_dict_from_url('https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth')
data = load_state_dict_from_url('https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth')
model.load_state_dict(data)

for i in data.keys():
    print(i)