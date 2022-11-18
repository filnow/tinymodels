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
			
			self.features = {}
			self.classifier = {}
							
	def forward(self, x: Tensor) -> Tensor:
			
			for i in range(16):
				if i in [1,3,7,11,15]:
					x = x.conv2d(*self.features[i], padding=1).relu().max_pool2d(kernel_size=(2,2))
				else:
					x = x.conv2d(*self.features[i], padding=1).relu()
			
			x = x.avg_pool2d(kernel_size=(1,1))
			
			x = x.flatten(1)

			x = x.linear(*self.classifier[0]).relu().dropout(0.5)
			x = x.linear(*self.classifier[1]).relu().dropout(0.5)
			x = x.linear(*self.classifier[2])

			return x.softmax()
		
	def fake_load(self):

		data = load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', progress=True)

		features_weight = {}
		classifier_weight = {}
		
		features_bias = {}
		classifier_bias = {}

		for k,v in data.items():

			if 'features' in k:    
				if 'weight' in k:
					features_weight[k] = Tensor.glorot_uniform(*v.detach().numpy().shape).assign(v.detach().numpy())
				else:
					features_bias[k] = Tensor.zeros(*v.detach().numpy().shape).assign(v.detach().numpy())
			else:  
				if 'weight' in k:
					classifier_weight[k] = Tensor.glorot_uniform(*v.detach().numpy().T.shape).assign(v.detach().numpy().T)
				else:
					classifier_bias[k] = Tensor.zeros(*v.detach().numpy().shape).assign(v.detach().numpy())

		self.features = tuple(zip(self.features_weight.values(), self.features_bias.values()))
		self.classifier = tuple(zip(self.classifier_weight.values(), self.classifier_bias.values()))

model = VGG19()
model.fake_load()

response = requests.get(sys.argv[1])
labels = requests.get('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt').text.split('\n')

img = Image.open(BytesIO(response.content))
out, _ = infer(model, img)
print(np.argmax(out.data), np.max(out.data)*100, labels[np.argmax(out.data)])
