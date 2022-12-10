import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from utils import class_img

class MBConv(nn.Module):
    def __init__(self, inch, outch, strd) -> None:
        super().__init__()
        
        self._expand_conv = nn.Conv2d(inch, outch, kernel_size=3, stride=strd)
        self._bn0 = nn.BatchNorm2d(outch)
        self._depthwise_conv = nn.Conv2d(inch, outch, kernel_size=3, stride=strd, groups=10)
        self._bn1 = nn.BatchNorm2d(outch)
        self._se_reduce = reduce
        self._se_expand = expand
        self._project_conv = nn.Conv2d(inch, outch, kernel_size=3, stride=strd)
        self._bn2 = nn.BatchNorm2d(outch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class EfficientNET(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self._conv_stem = nn.Conv2d(3, 32, kernel_size=3, stride=3, bias=False)
        self._bn0 = nn.BatchNorm2d(32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = F.relu(self._bn0(self._conv_stem(x)), inplace=True)
        
        return x





model = EfficientNET()

#data = load_state_dict_from_url('https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth')
data = load_state_dict_from_url('https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth')
#model.load_state_dict(data)

for i in data.keys():
    print(i)