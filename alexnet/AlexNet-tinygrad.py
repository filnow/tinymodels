from tinygrad.tensor import Tensor 
import torch
from torch.hub import load_state_dict_from_url
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO
from extra.utils import get_parameters
class AlexNet():

    def __init__(self):
        
        self.conv0 = (Tensor.glorot_uniform(64, 3, 11, 11), Tensor.zeros(64))
        self.conv3 = (Tensor.glorot_uniform(192, 64, 5, 5), Tensor.zeros(192))
        self.conv6 = (Tensor.glorot_uniform(384, 192, 3, 3), Tensor.zeros(384))
        self.conv8 = (Tensor.glorot_uniform(256, 384, 3, 3), Tensor.zeros(256))
        self.conv10 = (Tensor.glorot_uniform(256, 256, 3, 3), Tensor.zeros(256))

        self.ll1 = (Tensor.uniform(4096, 9216), Tensor.zeros(4096))
        self.ll4 = (Tensor.uniform(4096, 4096), Tensor.zeros(4096))
        self.ll6 = (Tensor.uniform(1000, 4096), Tensor.zeros(1000))
    
    #NOTE should use relu not leakyrelu
    def forward(self, x: Tensor) -> Tensor:
         
        x = x.conv2d(*self.conv0, padding=2, stride=4).relu().max_pool2d()
        x = x.conv2d(*self.conv3, padding=2).relu().max_pool2d()
        x = x.conv2d(*self.conv6, padding=1).relu()
        x = x.conv2d(*self.conv8, padding=1).relu()
        x = x.conv2d(*self.conv10, padding=1).relu().max_pool2d()
        
        x = x.avg_pool2d(kernel_size=(1,1))
 
        x = x.flatten(1)
        x = x.dropout(0.5)
        
        x = x.linear(self.ll1[0].transpose(), self.ll1[1]).relu()
        x = x.dropout(0.5)
        
        x = x.linear(*self.ll4).relu()
        x = x.linear(self.ll6[0].transpose(), self.ll6[1])

        return x
      
    def fake_load(m):

      dat = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth', progress=True)
              
      m.conv0[0].assign(dat['features.0.weight'].detach().numpy())
      m.conv0[1].assign(dat['features.0.bias'].detach().numpy())
      m.conv3[0].assign(dat['features.3.weight'].detach().numpy())
      m.conv3[1].assign(dat['features.3.bias'].detach().numpy())
      m.conv6[0].assign(dat['features.6.weight'].detach().numpy())
      m.conv6[1].assign(dat['features.6.bias'].detach().numpy())
      m.conv8[0].assign(dat['features.8.weight'].detach().numpy())
      m.conv8[1].assign(dat['features.8.bias'].detach().numpy())
      m.conv10[0].assign(dat['features.10.weight'].detach().numpy())
      m.conv10[1].assign(dat['features.10.bias'].detach().numpy())
    
      m.ll1[0].assign(dat['classifier.1.weight'].detach().numpy())
      m.ll1[1].assign(dat['classifier.1.bias'].detach().numpy())
      m.ll4[0].assign(dat['classifier.4.weight'].detach().numpy())
      m.ll4[1].assign(dat['classifier.4.bias'].detach().numpy())
      m.ll6[0].assign(dat['classifier.6.weight'].detach().numpy())
      m.ll6[1].assign(dat['classifier.6.bias'].detach().numpy())

m = AlexNet()
m.fake_load()


transform= transforms.Compose([            
 
 transforms.Resize(256),                    
 transforms.CenterCrop(224),                
 transforms.ToTensor(),                     
 transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],                
 std=[0.229, 0.224, 0.225]                  
 
 )])

#'https://raw.githubusercontent.com/srirammanikumar/DogBreedClassifier/master/images/Labrador_retriever_06457.jpg'
response = requests.get('https://raw.githubusercontent.com/filnow/tinymodels/main/data/dog3.jpg')
img = Image.open(BytesIO(response.content))
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)
out = m.forward(Tensor(batch_t.detach().numpy()))
labels = requests.get('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt').text.split('\n')
#index = int(Tensor.max(out).numpy())
_, index = torch.max(torch.tensor(out.numpy()), 1)
#percentage = out.softmax()[0] * 100
percentage = torch.nn.functional.softmax(torch.tensor(out.numpy()), dim=1)[0] * 100

print(labels[index[0]], percentage[index[0]].item())
