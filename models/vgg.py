import torch
import torch.nn as nn
from typing import List


class VGG(nn.Module):
    def __init__(self):
        super().__init__()

        layers: List[nn.Module] = []

        inch, outch = 3, 64

        cfgs = [2,2,4,4,4]      # VGG16 sum(cfgs)

        for i in range(len(cfgs)):

            for _ in range(cfgs[i]):

                layers.extend([nn.Conv2d(inch, outch, kernel_size=3, padding=1), nn.ReLU(inplace=True)])
                inch = outch

            outch = outch*2 if i < (len(cfgs)-2) else outch

            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*layers)
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
 
        return x
