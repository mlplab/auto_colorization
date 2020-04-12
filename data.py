# coding: utf-8


import os
import numpy as np
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
        img = Image.open(os.path.join(
            self.img_path, self.data[idx])).convert('RGB')
        if self.transform is not None:
            label = self.transform(img)
        else:
            label = img
        inputs = self.transform2gray(label)
        label = torchvision.transforms.ToTensor()(label)
        return inputs, label

    def __len__(self):
        return self.length


class YCbCrDataset(torch.utils.data.Dataset):

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
        self.weight_tensor = torch.as_tensor(
            np.array([[0.299, 0.587, 0.114],
                      [0.5, -0.418688, -0.081312],
                      [-0.168736, -0.331264, 0.5]], np.float32).reshape(3, 3, 1, 1)
        )

    def __getitem__(self, idx):
        img = Image.open(os.path.join(
            self.img_path, self.data[idx])).convert('RGB')
        img = self.__rgb2ycbcr(img)
        if self.transform is not None:
            img = self.transform(img)
        # inputs = self.transform2gray(label)
        # label = torchvision.transforms.ToTensor()(label)
        inputs = img[:, 0].unsqueeze(1)
        labels = img[:, 1:]
        return inputs, labels

    def __len__(self):
        return self.length

    def __rgb2ycbcr(self, img):
        img = torchvision.transforms.ToTensor()(img).unsqueeze(0)
        trans_img = torch.nn.functional.conv2d(img, self.weight_tensor)
        trans_img *= 2.0
        trans_img[:, 0, :, :] -= 1.0
        return trans_img


if __name__ == '__main__':

    train_data = os.listdir('dataset/train')
    dataset = YCbCrDataset('dataset/train', train_data)
    transform2gray = torchvision.transforms.Compose([
        # torchvision.transforms.Grayscale(),
        torchvision.transforms.RandomCrop(224, 224),
        torchvision.transforms.ToTensor()
    ])
    for i in range(10, 20):
        print(dataset[i][0].shape, dataset[i][0].min(), dataset[i][0].max())
        print(dataset[i][1].shape, dataset[i][1].min(), dataset[i][1].max())
        print()
