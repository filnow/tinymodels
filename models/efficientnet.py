import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)


class MBConv(nn.Module):
    def __init__(self, inch: int, k: int, s:int , project_out: int, dilation: int = 1, block0: bool = True) -> None:
        super().__init__()
        
        self.block0 = block0
        out = int(inch * 0.25)
        outch = inch

        if s == 2:
            pad = ((k-1)//2-1, (k-1)//2)
        else:
            pad = ((k-1)//2, (k-1)//2)
        
        if block0:
            outch = inch*6
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

        self.cfgs = [
            # t, c, n, s, k
            [16,  24, 2, 2, 3],
            [24,  40, 2, 2, 5],
            [40,  80, 3, 2, 3],
            [80,  112, 3, 1, 5],
            [112,  192, 4, 1, 5],
            [192, 320, 1, 2, 3]
        ]

        blocks = [MBConv(32,3,1,16, block0=False)]

        for t,c,n,s,k in self.cfgs:
            for i in range(n):
                if i == 0:
                    blocks.append(MBConv(t,k,s,c))
                else:
                    blocks.append(MBConv(c,k,1,c))

        self._blocks = nn.Sequential(*blocks)
 
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