import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from utils import class_img

class MBConv(nn.Module):
    def __init__(self, inch, outch, k, s, block0=True) -> None:
        super().__init__()
        
        out = inch * 0.25
        out_project = outch * 0.25


        if block0:
            self._expand_conv = nn.Conv2d(inch, outch, kernel_size=1, stride=s, bias=False)
            self._bn0 = nn.BatchNorm2d(outch)
        
        self._depthwise_conv = nn.Conv2d(1, outch,  kernel_size=5, stride=s,  groups=1, bias=False)
        self._bn1 = nn.BatchNorm2d(outch)
        self._se_reduce = nn.Conv2d(outch, out, kernel_size=1, stride=s)
        self._se_expand = nn.Conv2d(out, outch, kernel_size=1, stride=s)
        self._project_conv = nn.Conv2d(outch, out_project, kernel_size=1, stride=s, bias=False)
        self._bn2 = nn.BatchNorm2d(out_project)                   
        
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
            
            MBConv(1,32,3,1, block0=False),
            MBConv(16,96,1,3),
            MBConv(24,144,1,1),
            MBConv(24,144,1,1),
            MBConv(40,240,1,1),
            MBConv(40,240,1,1),
            MBConv(80,480,1,1),
            MBConv(80,480,1,1),
            MBConv(80,480,1,1),
            MBConv(112,672,1,1),
            MBConv(112,672,1,1),
            MBConv(112,672,1,1),
            MBConv(192,1152,1,1),
            MBConv(192,1152,1,1),
            MBConv(192,1152,1,1),
            MBConv(192,1152,1,1)

        )

        self._conv_head = nn.Conv2d(320, 1280, kernel_size=1, stride=3, bias=False)
        self._bn1 = nn.BatchNorm2d(1280)
        self._fc = nn.Linear(1280, 1000)
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
    print(i, data[i].shape)