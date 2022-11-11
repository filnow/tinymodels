import torch.nn as nn
import torch
from tinygrad.tensor import Tensor 

class MaxPool2d:
  def __init__(self, kernel_size, stride):
    if isinstance(kernel_size, int): self.kernel_size = (kernel_size, kernel_size)
    else: self.kernel_size = kernel_size
    self.stride = stride if (stride is not None) else kernel_size
  
  def __repr__(self):
    return f"MaxPool2d(kernel_size={self.kernel_size!r}, stride={self.stride!r})"
  
  def __call__(self, input):
    # TODO: Implement strided max_pool2d, and maxpool2d for 3d inputs
    return input.max_pool2d(kernel_size=self.kernel_size)

m = nn.MaxPool2d(3, stride=2)
n = tMaxPool2d(3,2)
input = Tensor(torch.randn(20, 16, 50, 32))
output = n(input)




maxpool = Tensor.max_pool2d(Tensor(torch.randn(20, 16, 50, 32)), kernel_size=3)


print(output)