# coding: utf-8


import os
import numpy as np
from tqdm import tqdm, trange
from time import time
from collections import OrderedDict
import torch
import torchvision
from torchsummary import summary
from model.encoder_decoder import CAE
from data import AutoColorDataset
from utils import Draw_Output, ModelCheckPoint, psnr


class Trainer(object):

    def __init__(self, model, criterion, optimizer, device='cpu', callbacks=None):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=.2,
                                                                    patience=2,
                                                                    verbose=True,
                                                                    min_lr=1e-8)
        self.device = device
        self.callbacks = callbacks

    def train(self, epochs, train_dataloader, val_dataloader, init_epoch=None):

        if init_epoch is None:
            init_epoch = 0
        elif isinstance(init_epoch, int):
            assert 'Please enter int to init_epochs'
        

        for epoch in range(init_epoch, epochs):
            self.model.train()
            mode = 'Train'
            train_loss = []
            val_loss = []
            with tqdm(train_dataloader, desc=f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}', ncols=None, unit='step') as pbar:
                for i, (inputs, labels) in enumerate(pbar):
                    inputs, labels = self.__trans_data(inputs, labels)
                    loss = self.__step(inputs, labels)
                    train_loss.append(loss.item())
                    psnr_show = psnr(loss)
                    self.__step_show(pbar, mode, epoch, loss, psnr_show)
                    torch.cuda.empty_cache()
            mode = 'Val'
            self.model.eval()
            with tqdm(val_dataloader, desc=f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}', ncols=None, unit='step') as pbar:
                for i, (inputs, labels) in enumerate(pbar):
                    inputs, labels = self.__trans_data(inputs, labels)
                    with torch.no_grad():
                        loss = self.__step(inputs, labels, train=False)
                    val_loss.append(loss.item())
                    psnr_show = psnr(loss)
                    self.__step_show(pbar, mode, epoch, loss, psnr_show)
                    torch.cuda.empty_cache()
            train_loss = np.mean(train_loss)
            val_loss = np.mean(val_loss)
            if self.callbacks:
                for callback in self.callbacks:
                    callback.callback(self.model, epoch, loss=train_loss, val_loss=val_loss, save=True, device=self.device)
            _, columns = os.popen('stty size', 'r').read().split()
            self.scheduler.step(val_loss)
            print('-' * int(columns))

    def __trans_data(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        return inputs, labels

    def __step(self, inputs, labels, train=True):
        if train is True:
            self.optimizer.zero_grad()
        output = self.model(inputs)
        loss = self.criterion(output, labels)
        if train is True:
            loss.backward()
            self.optimizer.step()
        return loss

    def __step_show(self, pbar, mode, epoch, loss, psnr_show):
        if self.device is 'cuda':
            pbar.set_postfix(
                    OrderedDict(
                        Loss=f'{loss:.7f}',
                        PSNR=f'{psnr_show:.7f}',
                        Allocate=f'{torch.cuda.memory_allocated(0) / 1024 ** 3:.3f}GB',
                        Cache=f'{torch.cuda.memory_cached(0) / 1024 ** 3:.3f}GB'
                        )
                    )
        elif self.device is 'cpu':
            pbar.set_postfix(
                    OrderedDict(
                        Loss=f'{loss:.7f};',
                        PSNR=f'{psnr_show:.7f}'
                        )
                    )
        return self


if __name__ == '__main__':

    resize = 256
    crop = 96
    batch_size = 2
    epochs = 5
    data_len = batch_size * 10

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_path = 'dataset/'
    train_path = os.path.join(img_path, 'train')
    test_path = os.path.join(img_path, 'test')
    # drive_path = '/content/drive/My Drive/auto_colorization/'

    train_list = os.listdir(train_path)
    test_list = os.listdir(test_path)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((resize, resize)),
        torchvision.transforms.RandomCrop(crop)
    ])
    train_dataset = AutoColorDataset(train_path, train_list[:data_len], transform)
    test_dataset = AutoColorDataset(test_path, test_list[:data_len], transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = CAE([64, 128], num_layer=2).to(device)
    # summary(model, (1, 96, 96))
    criterion = torch.nn.MSELoss().to(device)
    param = list(model.parameters())
    optimizer = torch.optim.Adam(lr=1e-3, params=param)
    draw_cb = Draw_Output(test_path, test_list[:9], nrow=3, save_path='output', verbose=False)
    ckpt_cb = ModelCheckPoint('ckpt', 'CAE')

    trainer = Trainer(model, criterion, optimizer, device=device, callbacks=[draw_cb, ckpt_cb])
    trainer.train(epochs, train_dataloader, test_dataloader, init_epoch=3)
