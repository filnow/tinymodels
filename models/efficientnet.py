import torch
import torch.nn as nn
import torch.nn.functional as F

def swish(x):
    return x * torch.sigmoid(x)


class MBConv(nn.Module):
    def __init__(self, inch: int, outch: int, k: int, s:int , project_out: int, dilation: int = 1, block0: bool = True) -> None:
        super().__init__()
        
        self.block0 = block0
        out = int(inch * 0.25)

        if s == 2:
            pad = [(k-1)//2-1, (k-1)//2]
        else:
            pad = [(k-1)//2]
        
        if block0:
            self._expand_conv = nn.Conv2d(inch, outch, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(outch)
        
        self._depthwise_conv = nn.Conv2d(outch, outch, kernel_size=k, stride=s, padding=pad, dilation=dilation, groups=outch, bias=False)
        self._bn1 = nn.BatchNorm2d(outch)
        
        self._se_reduce = nn.Conv2d(outch, out, kernel_size=1)
        self._se_expand = nn.Conv2d(out, outch, kernel_size=1)
        
        self._project_conv = nn.Conv2d(outch, project_out, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(project_out)                   
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        
        ident = x
        
        if self.block0:
            x = swish(self._bn0(self._expand_conv(x)))
        
        x = swish(self._bn1(self._depthwise_conv(x)))
        
        x_squeezed = F.adaptive_avg_pool2d(x, 1)
        x_squeezed = swish(self._se_reduce(x_squeezed))
        x_squeezed = self._se_expand(x_squeezed)
        x  = torch.sigmoid(x_squeezed) * x
        
        x = self._bn2(self._project_conv(x))
        
        if x.shape == ident.shape:
            x+=ident

        return x

class EfficientNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self._conv_stem = nn.Conv2d(3, 32, kernel_size=3, stride=2,padding=1, bias=False)
        self._bn0 = nn.BatchNorm2d(32)

        self._blocks = nn.Sequential(
            
            MBConv(32,32,3,1,16, block0=False),
            MBConv(16,96,3,2,24),
            MBConv(24,144,3,1,24),
            MBConv(24,144,5,2,40),
            MBConv(40,240,5,1,40),
            MBConv(40,240,3,2,80),
            MBConv(80,480,3,1,80),
            MBConv(80,480,3,1,80),
            MBConv(80,480,5,1,112),
            MBConv(112,672,5,1,112),
            MBConv(112,672,5,1,112),
            MBConv(112,672,5,1,192),
            MBConv(192,1152,5,1,192),
            MBConv(192,1152,5,1,192),
            MBConv(192,1152,5,1,192),
            MBConv(192,1152,3,2,320)

        )

        self._conv_head = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(1280)
        self._fc = nn.Linear(1280, 1000)
        self.dropout = nn.Dropout()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = swish(self._bn0(self._conv_stem(x)))
        x = self._blocks(x)
        x = swish(self._bn1(self._conv_head(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self._fc(self.dropout(x))

        return x