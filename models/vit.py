import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(768, 3072)
        self.linear_2 = nn.Linear(3072, 768)

        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = F.gelu(self.linear_1(self.dropout(x)), inplace=True)
        x = F.gelu(self.linear_2(x), inplace=True)

        return x


class EncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(768)
        self.self_attention = nn.MultiheadAttention(768, 768, batch_first=True)
        self.ln_2 = nn.LayerNorm(768)
        self.mlp = MLP()
        
        self.dropout = nn.Dropout()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        ident = x
        
        x = self.ln_1(x)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        
        x += ident
        
        y = self.ln_2(x)
        y = self.mlp(y)

        return x + y


class Layers(nn.Module):
    def __init__(self, num: int) -> None:
        super().__init__()

        for i in range(num):
            setattr(self, f"encoder_layer_{i}", EncoderLayer())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.encoder_layer_1(x)


class Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()        
        self.dropout = nn.Dropout()
        self.head = nn.Linear(768,1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.head(self.dropout(x))


class ViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.class_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.conv_proj = nn.Conv2d(3, 768, kernel_size=16)

        self.encoder = nn.Sequential(

            OrderedDict(
                [   
                    ('pos_embedding', nn.ParameterList(torch.empty(1,197,768).normal_(std=0.02))),
                    ('layers', Layers(12)),
                    ('ln', nn.LayerNorm(768))
                ]
            )
        )
        
        self.heads = Head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_proj(x)
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x

data = load_state_dict_from_url('https://download.pytorch.org/models/vit_b_16-c867db91.pth')

model = ViT()
model.load_state_dict(data)
model.eval()

for i in data.keys():
    print(i, data[i].shape)