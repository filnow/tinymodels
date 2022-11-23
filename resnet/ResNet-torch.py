from collections import OrderedDict
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.hub import load_state_dict_from_url



class IdentityBlock(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels,in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = torch.add(self.attention, x).relu()
        return x

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, strd) -> None:
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=1, stride=strd),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,in_channels,kernel_size=3, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
            
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=strd),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = torch.add(self.layer0, self.downsample).relu()
        return x

class ResNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer = nn.Sequential(

            ConvBlock(64, 256,1),
            IdentityBlock(64,256),
            IdentityBlock(64,256),
            ConvBlock(128, 512,2),
            IdentityBlock(128,512),
            IdentityBlock(128,512),
            IdentityBlock(128,512),
            ConvBlock(256, 1024,2),
            IdentityBlock(256,1024),
            IdentityBlock(256,1024),
            IdentityBlock(256,1024),
            IdentityBlock(256,1024),
            IdentityBlock(256,1024),
            ConvBlock(512, 2048,2),
            IdentityBlock(512,2048),
            IdentityBlock(512,2048)
            
        )        
        
        self.classifier = nn.Sequential(

            nn.Dropout(),
            nn.Linear(512,1000),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

#data  = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth')
data  = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-0676ba61.pth')
model = ResNet()
model.load_state_dict(data)

#for i in data.keys():
    #print(i , data[i].detach().numpy().shape, '\n')