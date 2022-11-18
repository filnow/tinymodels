import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.hub import load_state_dict_from_url


class AttentionBlock:

    def __init__(self) -> None:
        pass


class ResNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64)
        )

        self.layer = nn.Sequential(

            AttentionBlock(),
            AttentionBlock(),
            AttentionBlock(),
            AttentionBlock()
        )        
        
        self.classifier = nn.Sequential(

            nn.Dropout(),
            nn.Linear(512,1000),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.features(x)
        x = self.layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

data  = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth')


for i in data.keys():
    print(i , data[i].detach().numpy().shape, '\n')