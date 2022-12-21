import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from utils import class_img

class MBConv(nn.Module):
    def __init__(self, inch: int, outch: int, k: int, s:int , project_out: int, block0: bool = True) -> None:
        super().__init__()
        
        self.block0 = block0
        out = int(inch * 0.25)
        
        if block0:
            self._expand_conv = nn.Conv2d(inch, outch, kernel_size=1, stride=s,  bias=False)
            self._bn0 = nn.BatchNorm2d(outch)
        
        self._depthwise_conv = nn.Conv2d(outch, outch, kernel_size=k, stride=s, groups=outch, bias=False)
        self._bn1 = nn.BatchNorm2d(outch)
        self._se_reduce = nn.Conv2d(outch, out, kernel_size=1, stride=s)
        self._se_expand = nn.Conv2d(out, outch, kernel_size=1, stride=s)
        self._project_conv = nn.Conv2d(outch, project_out, kernel_size=1, stride=s, bias=False)
        self._bn2 = nn.BatchNorm2d(project_out)                   
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        
        ident = x
        print(ident.shape)
        if self.block0:
            x = F.silu(self._bn0(self._expand_conv(x)), inplace=True)
        x = F.silu(self._bn1(self._depthwise_conv(x)), inplace=True)
        x = self.avgpool(x)
        x = F.silu(self._se_reduce(x), inplace=True)
        x = F.silu(self._se_expand(x), inplace=True)
        x = F.silu(self._bn2(self._project_conv(x)), inplace=True)
        print(x.shape)
        #x += ident
        return x

class EfficientNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self._conv_stem = nn.Conv2d(3, 32, kernel_size=3, stride=1, bias=False)
        self._bn0 = nn.BatchNorm2d(32)

        self._blocks = nn.Sequential(
            
            MBConv(32,32,3,1, 16, block0=False),
            MBConv(16,96,3,2, 24),
            MBConv(24,144,3,2, 24),
            MBConv(24,144,5,2, 40),
            MBConv(40,240,5,2, 40),
            MBConv(40,240,3,2, 80),
            MBConv(80,480,3,2, 80),
            MBConv(80,480,3,1, 80),
            MBConv(80,480,5,1, 112),
            MBConv(112,672,5,2, 112),
            MBConv(112,672,5,2, 112),
            MBConv(112,672,5,2, 192),
            MBConv(192,1152,5,1, 192),
            MBConv(192,1152,5,1, 192),
            MBConv(192,1152,5,1, 192),
            MBConv(192,1152,3,1, 320)

        )

        self._conv_head = nn.Conv2d(320, 1280, kernel_size=1, stride=1, bias=False)
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


model = EfficientNet()

#data = load_state_dict_from_url('https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth')
data = load_state_dict_from_url('https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth')
model.load_state_dict(data)
model.eval()

class_name, precentage = class_img(model, './images/Labrador_retriever_06457.jpg')

print(class_name)
#for i in data.keys():
    #print(i, data[i].shape)