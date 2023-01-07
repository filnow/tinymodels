import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F


class Layer2dNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)

class Permute(nn.Module):

    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return torch.permute(x, self.dims)


class BootleNeck(nn.Module):
    
    def __init__(self, inch: int, outch: int) -> None:
        super().__init__()
        self.layer_scale = nn.Parameter(torch.FloatTensor(inch, 1, 1))
        
        self.block = nn.Sequential(

            nn.Conv2d(inch, inch, kernel_size=7, padding = 3, groups=inch),
            Permute([0,2,3,1]),     
            nn.LayerNorm(inch),
            nn.Linear(inch, outch),
            nn.GELU(),
            nn.Linear(outch,inch),
            Permute([0,3,1,2]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = self.layer_scale * self.block(x) 
        
        return x + out


class ConvNeXt(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        layers: List[nn.Module] = []

        #patchify layer
        layers.append(
            nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=4, stride=4),
                Layer2dNorm(96)
            )
        )

        inch, outch = 96, 384
        cfgs = [3,3,9,3]

        for i in range(len(cfgs)):

            block: List[nn.Module] = []

            for _ in range(cfgs[i]):

                block.append(
                    BootleNeck(inch, outch)
                )

            layers.append(nn.Sequential(*block))
            
            if i < (len(cfgs)-1):
                
                layers.append(
                    nn.Sequential(
                        Layer2dNorm(inch),
                        nn.Conv2d(inch, outch//2, kernel_size=2, stride=2)
                    )
                )
                
            inch, outch = inch*2, outch*2

        self.features = nn.Sequential(*layers)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.classifier = nn.Sequential(
            Layer2dNorm(768), 
            nn.Flatten(1),
            nn.Linear(768,1000)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        
        return x
