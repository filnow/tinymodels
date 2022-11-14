#https://download.pytorch.org/models/vgg16-397923af.pth
# features "https://download.pytorch.org/models/vgg16_features-amdegroot-88682ab5.pth"
#https://arxiv.org/abs/1505.06798.pdf
import torch 
import torch.nn as nn
from torch.hub import load_state_dict_from_url




data = load_state_dict_from_url("https://download.pytorch.org/models/vgg16_features-amdegroot-88682ab5.pth")
print(data.keys())