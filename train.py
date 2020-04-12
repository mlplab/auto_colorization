# coding: utf-8


import os
import numpy as np
import torch
import torchvision
from torchsummary import summary
from encoder_decoder import CAE
from unet import UNet
from trainer import Trainer
from data import AutoColorDataset
from utils import Draw_Output, ModelCheckPoint


resize = 256
crop = 96
batch_size = 125
# data_len = batch_size * 10
data_len = None
epochs = 100


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


img_path = 'dataset/'
train_path = os.path.join(img_path, 'train')
test_path = os.path.join(img_path, 'test')
# drive_path = '/content/drive/My Drive/auto_colorization/'


train_list = os.listdir(train_path)
test_list = os.listdir(test_path)
# output_list = np.random.choice(test_list, 64)
output_list = test_list[:64]
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((resize, resize)),
    torchvision.transforms.RandomCrop(crop)
])
train_dataset = AutoColorDataset(train_path, train_list[:data_len], transform)
test_dataset = AutoColorDataset(test_path, test_list[:data_len], transform)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# model = CAE([64, 128, 256]).to(device)
model = UNet(1, 3).to(device)
summary(model, (1, 96, 96))
criterion = torch.nn.MSELoss().to(device)
param = list(model.parameters())
optimizer = torch.optim.Adam(lr=1e-3, params=param)
draw_cb = Draw_Output(test_path, output_list, nrow=8,
                      save_path='output_unet', verbose=False)
ckpt_cb = ModelCheckPoint('ckpt_unet', 'UNet')

trainer = Trainer(model, criterion, optimizer, device=device,
                  callbacks=[draw_cb, ckpt_cb])
trainer.train(epochs, train_dataloader, test_dataloader)
