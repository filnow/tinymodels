import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.hub import load_state_dict_from_url
import requests
from io import BytesIO
import sys

class Inception(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(3, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b1 = self.branch1(x.clone())
        b2 = self.branch2(x.clone())
        b3 = self.branch3(x.clone())
        b4 = self.branch4(x.clone())

        x = torch.cat((b1,b2,b3,b4), 0)

        return x

class GoogleNET(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.conv1  = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(192),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.inception3a = Inception()
        self.inception3b = Inception()
        
        self.inception4a = Inception()
        self.inception4b = Inception()
        self.inception4c = Inception()
        self.inception4d = Inception()
        self.inception4e = Inception()
        
        self.inception5a = Inception()
        self.inception5b = Inception()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024,1000)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x



model = GoogleNET()
data = load_state_dict_from_url('https://download.pytorch.org/models/googlenet-1378be20.pth')
model.load_state_dict(data)

#https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py