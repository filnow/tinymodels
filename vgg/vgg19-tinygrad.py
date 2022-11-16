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
        
        self.features = (

            (Tensor.glorot_uniform(64, 3, 3, 3), Tensor.zeros(64)),
            (Tensor.glorot_uniform(64, 64, 3, 3), Tensor.zeros(64)),
            (Tensor.glorot_uniform(128, 64, 3, 3), Tensor.zeros(128)),
            (Tensor.glorot_uniform(128, 128, 3, 3), Tensor.zeros(128)),
            (Tensor.glorot_uniform(256, 128, 3, 3), Tensor.zeros(256)),
            (Tensor.glorot_uniform(256, 256, 3, 3), Tensor.zeros(256)),
            (Tensor.glorot_uniform(256, 256, 3, 3), Tensor.zeros(256)),
            (Tensor.glorot_uniform(256, 256, 3, 3), Tensor.zeros(256)),
            (Tensor.glorot_uniform(512, 256, 3, 3), Tensor.zeros(512)),
            (Tensor.glorot_uniform(512, 512, 3, 3), Tensor.zeros(512)),
            (Tensor.glorot_uniform(512, 512, 3, 3), Tensor.zeros(512)),
            (Tensor.glorot_uniform(512, 512, 3, 3), Tensor.zeros(512)),
            (Tensor.glorot_uniform(512, 512, 3, 3), Tensor.zeros(512)),
            (Tensor.glorot_uniform(512, 512, 3, 3), Tensor.zeros(512)),
            (Tensor.glorot_uniform(512, 512, 3, 3), Tensor.zeros(512)),
            (Tensor.glorot_uniform(512, 512, 3, 3), Tensor.zeros(512))

        )

        self.classifier = (
            
            (Tensor.uniform(4096, 25088), Tensor.zeros(4096)),
            (Tensor.uniform(4096, 4096), Tensor.zeros(4096)),
            (Tensor.uniform(1000, 4096), Tensor.zeros(1000))
    
        )
        
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
      
    def fake_load(self):

        data = load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', progress=True)
        
        for i in range(len(self.features)):
            for j in data:
                if 'weight' in j:                    
                    self.features[i][0].assign(data[j].detach().numpy())
                else:
                    self.features[i][1].assign(data[j].detach().numpy())
        
        for i in range(len(self.classifier)):
            for j in data:
                if 'weight' in j:                    
                    self.features[i][0].assign(data[j].detach().numpy())
                else:
                    self.features[i][1].assign(data[j].detach().numpy())

m = AlexNet()
m.fake_load()

'''
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
'''
