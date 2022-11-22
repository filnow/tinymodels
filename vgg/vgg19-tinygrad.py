from tinygrad.tensor import Tensor 
from torch.hub import load_state_dict_from_url
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from utils import infer
import sys

class VGG19():

	def __init__(self):
			
			self.wb = []
							
	def forward(self, x: Tensor) -> Tensor:
			
			for i in range(16):
				if i in [1,3,7,11,15]:
					x = x.conv2d(*self.wb[i], padding=1).relu().max_pool2d(kernel_size=(2,2))
				else:
					x = x.conv2d(*self.wb[i], padding=1).relu()
			
			x = x.avg_pool2d(kernel_size=(1,1)).flatten(1)
			
			x = x.linear(self.wb[16][0].transpose(), self.wb[16][1]).relu().dropout(0.5)
			x = x.linear(self.wb[17][0].transpose(), self.wb[17][1]).relu().dropout(0.5)
			x = x.linear(self.wb[18][0].transpose(), self.wb[18][1]).softmax()

			return x
		
	def fake_load(self):

		data = load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', progress=True)

		dk = list(data.keys())
		
		for i in range(len(dk)):
			if i%2 == 0:
				self.wb.append((
					Tensor.glorot_uniform(*data[dk[i]].detach().numpy().shape).assign(data[dk[i]].detach().numpy()),
					Tensor.zeros(*data[dk[i+1]].detach().numpy().shape).assign(data[dk[i+1]].detach().numpy())
				))
		
model = VGG19()
model.fake_load()

response = requests.get(sys.argv[1])
labels = requests.get('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt').text.split('\n')

img = Image.open(BytesIO(response.content))
out, _ = infer(model, img)
print(np.argmax(out.data), np.max(out.data)*100, labels[np.argmax(out.data)])
