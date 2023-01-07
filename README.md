# Motivation

The goal of this repo is to implement DL models for image classification in:

* [pytorch](https://github.com/pytorch/pytorch)

And to make a simple web app to display results of this models

Models are implemented from the papers with pretrained weights

tinymodels is just for learning how models actually work

# Architectures

* [AlexNet](https://arxiv.org/pdf/1404.5997.pdf) - 2014     &#x2714;

* [GoogleNET](https://arxiv.org/pdf/1409.4842.pdf) - 2014   &#x2714;

* [VGG](https://arxiv.org/pdf/1505.06798.pdf) - 2015        &#x2714;

* [InceptionV3](https://arxiv.org/pdf/1512.00567.pdf) - 2015    &#x2714;

* [ResNet](https://arxiv.org/pdf/1704.06904.pdf) - 2017     &#x2714;
    
* [DenseNet](https://arxiv.org/pdf/1608.06993.pdf) - 2018   &#x2714;

* [MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf) - 2019    &#x2714;

* [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf) - 2020   &#x2714;

* [ViT](https://arxiv.org/pdf/2010.11929.pdf) - 2021    &#x2714;

* [SwinTransformer](https://arxiv.org/pdf/2103.14030.pdf) - 2021 

* [ConvNeXt](https://arxiv.org/pdf/2201.03545v2.pdf) - 2022     &#x2714;

# Running tinymodels

On default app will run on port 3000.
First run will take few minutes to download pretrained weights.

```bash

git clone https://github.com/filnow/tinymodels.git
cd tinymodels
pip install -r requirements.txt
python3 app.py

```