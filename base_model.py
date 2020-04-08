#coding: utf-8


import torch
import torchvision


class CNN_Block(torch.nn.Module):
    
    def __init__(self, in_feature, out_feature, kernel=3, pool=True, num_layer=3):
        super(CNN_Block, self).__init__()
        layers = []
        features = [in_feature] + [out_feature for _ in range(num_layer)]
        for i in range(len(features) - 1):
            layers.append(torch.nn.Conv2d(features[i], features[i + 1], kernel_size=kernel, padding=1))
            layers.append(torch.nn.BatchNorm2d(features[i + 1]))
            layers.append(torch.nn.ReLU())
        if pool is True:
            layers.append(torch.nn.MaxPool2d(2, 2))
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

    
class CNN_Block_for_UNet(CNN_Block):
    
    def __init__(self, in_feature, out_feature, kernel=3, pool=True, num_layer=3):
        super(CNN_Block_for_UNet, self).__init__(in_feature, out_feature, kernel, pool, num_layer)
        layers = []
        features = [in_feature] + [out_feature for _ in range(num_layer)]
        if pool is True:
            layers.append(torch.nn.MaxPool2d(2, 2))
        for i in range(len(features) - 1):
            layers.append(torch.nn.Conv2d(features[i], features[i + 1], kernel_size=kernel, padding=1))
            layers.append(torch.nn.BatchNorm2d(features[i + 1]))
            layers.append(torch.nn.ReLU())
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)
    

class D_CNN_Block(torch.nn.Module):
    
    def __init__(self, in_feature, out_feature, kernel=3, num_layer=3):
        super(D_CNN_Block, self).__init__()
        layers = [torch.nn.ConvTranspose2d(in_feature, out_feature, kernel_size=2, stride=2)]
        features = [out_feature for _ in range(num_layer)]
        for i in range(len(features) - 1):
            layers.append(torch.nn.Conv2d(features[i], features[i + 1], kernel_size=kernel, padding=1))
            layers.append(torch.nn.BatchNorm2d(features[i + 1]))
            layers.append(torch.nn.ReLU())
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

