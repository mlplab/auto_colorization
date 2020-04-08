# coding: utf-8


import os
from PIL import Image
import torch
import torchvision


class AutoColorDataset(torch.utils.data.Dataset):
    
    def __init__(self, img_path, data, transform=None):
        # super(Flickr8kDataset, self).__init__()
        self.img_path = img_path
        self.data = data
        self.transform = transform
        self.transform2gray = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor()
        ])
        self.length = len(data)
        
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_path, self.data[idx])).convert('RGB')
        if self.transform is not None:
            label = self.transform(img)
        else:
            label = img
        inputs = self.transform2gray(label)
        label = torchvision.transforms.ToTensor()(label)
        return inputs, label
    
    def __len__(self):
        return self.length