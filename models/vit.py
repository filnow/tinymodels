import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from collections import OrderedDict
from utils import class_img


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(768, 3072)
        self.linear_2 = nn.Linear(3072, 768)

        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = F.gelu(self.linear_1(self.dropout(x)))
        x = F.gelu(self.linear_2(self.dropout(x)))

        return x


class EncoderBlock(nn.Module):
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


class Encoder(nn.Module):
    def __init__(self, num: int) -> None:
        super().__init__()
        layers = OrderedDict((f"encoder_layer_{i}", EncoderBlock()) for i in range(num))

        self.pos_embedding = nn.Parameter(torch.empty(1,197,768).normal_(std=0.02))
        self.layers = nn.Sequential(layers)
        self.ln = nn.LayerNorm(768)
        self.dropout = nn.Dropout()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x += self.pos_embedding
        return self.ln(self.layers(self.dropout(x)))


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

        self.patch_size = 16 #vit_b_16
        self.class_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.conv_proj = nn.Conv2d(3, 768, kernel_size=self.patch_size, stride=self.patch_size)

        self.encoder = Encoder(12)
        self.heads = Head()

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        
        n, _, h, w = x.shape
        n_h = h // self.patch_size
        n_w = w // self.patch_size

        x = self.conv_proj(x)

        x = x.reshape(n, 768, n_h * n_w).permute(0,2,1)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self._process_input(x)

        batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        x = self.encoder(x)
        
        x = x[:, 0]
        
        x = self.heads(x)

        return x

data = load_state_dict_from_url('https://download.pytorch.org/models/vit_b_16-c867db91.pth')

model = ViT()
model.load_state_dict(data)
model.eval()

print(class_img(model, '/home/filnow/fun/tinymodels/static/images/Labrador_retriever_06457.jpg'))
