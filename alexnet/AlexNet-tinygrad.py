from tinygrad.tensor import Tensor 
import tinygrad.nn as nn
from torch.hub import load_state_dict_from_url
from extra.utils import get_child
import torch.nn as torchnn
class MaxPool2d:
  def __init__(self, kernel_size, stride):
    if isinstance(kernel_size, int): self.kernel_size = (kernel_size, kernel_size)
    else: self.kernel_size = kernel_size
    self.stride = stride if (stride is not None) else kernel_size
  
  def __repr__(self):
    return f"MaxPool2d(kernel_size={self.kernel_size!r}, stride={self.stride!r})"
  
  def __call__(self, input):
    
    return input.max_pool2d(kernel_size=self.kernel_size)

class AlexNet():

    def __init__(self) -> None:
        
        self.features = (

            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=2),
            nn.Conv2d(384, 256, kernel_size=3, padding=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
            
        )
        
        self.l1 = nn.Linear(256 * 6 * 6, 4096)
        self.l2 = nn.Linear(4096, 4096)
        self.l3 = nn.Linear(4096, 1000)

        self.maxpool = MaxPool2d(3,2)
    
    def forward(self, x: Tensor) -> Tensor:

        x = self.maxpool(self.conv1(x)).relu()
        x = self.bn2(self.conv2(x)).relu()
        x = self.bn2(self.conv3(x)).relu()
        x = self.bn2(self.conv4(x)).relu()
        x = self.bn2(self.conv5(x)).relu()

        x = Tensor.flatten(x, 1)
        x = Tensor.dropout(x)
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        x = self.l3(x).relu()

        return x

    def load_from_pretrained(self, url):
    
        self.url = url

        from torch.hub import load_state_dict_from_url
        state_dict = load_state_dict_from_url(self.url, progress=True)
        for k, v in state_dict.items():
            
            obj = get_child(self, k)
            dat = v.detach().numpy().T if "fc.weight" in k else v.detach().numpy()

            assert obj.shape == dat.shape, (k, obj.shape, dat.shape)
            obj.assign(dat)

    def __call__(self, x): return self.forward(x)
           

model = AlexNet()
model.load_from_pretrained('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
model.eval