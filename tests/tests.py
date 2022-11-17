from tinygrad.tensor import Tensor 
from torch.hub import load_state_dict_from_url
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

np.random.seed(seed=1234)
data = load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', progress=True)
inputs = np.random.rand(64, 3, 3, 3)

w = Tensor.glorot_uniform(64,3,3,3).assign(data['features.0.weight'].detach().numpy())
b = Tensor.zeros(64).assign(data['features.0.bias'].detach().numpy())

it = torch.from_numpy(inputs.astype('double'))
it2 = Tensor(inputs)
output_tinygrad = it2.conv2d(w,b, padding=1).relu()
output_torch = F.conv2d(it, torch.from_numpy(data['features.0.weight'].detach().numpy().astype('double')), torch.from_numpy(data['features.0.bias'].detach().numpy().astype('double')), padding=1).relu()

print('tiny \n', output_tinygrad.data)
print('torch \n', output_torch)

