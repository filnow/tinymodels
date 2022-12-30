import torch
import torch.nn as nn
from typing import Type


class Block(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, strd: int = 1, sample: bool = False) -> None:
        super().__init__()        
        self.sample = sample
        
        self.conv1 = nn.Conv2d(in_channels, out_channels,kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3, stride=strd, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, 4 * out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4 * out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        if sample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, 4 * out_channels,kernel_size=1,stride=strd, bias=False),
                nn.BatchNorm2d(4 * out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        ident = x
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        if self.sample:
            ident = self.downsample(ident)
        
        x += ident
        x = self.relu(x)
       
        return x

class ResNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.in_ch = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(Block, 2, 64, 1)
        self.layer2 = self._make_layer(Block, 3, 128, 2)
        self.layer3 = self._make_layer(Block, 5, 256, 2)
        self.layer4 = self._make_layer(Block, 2, 512, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(512 * 4, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.relu(self.bn1(self.conv1(x)))
 
        x = self.maxpool(x)
      
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _make_layer(self, Block: Type[Block], num_Blocks: int, in_channels: int, stride: int) -> nn.Sequential:
        
        layers = [Block(self.in_ch,in_channels, strd=stride, sample=True)]
        
        self.in_ch = in_channels*4

        for _ in range(num_Blocks):
            layers.append(Block(self.in_ch,in_channels))
        
        return nn.Sequential(*layers)

