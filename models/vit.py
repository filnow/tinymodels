import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url





data = load_state_dict_from_url('https://download.pytorch.org/models/vit_b_16-c867db91.pth')


for i in data.keys():
    print(i, data[i].shape)