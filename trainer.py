# coding: utf-8


import os
import numpy as np
import datetime
from time import time
import torch
import torchvision
from torchsummary import summary
from encoder_decoder import CAE
from data import AutoColorDataset
from utils import Draw_Output, ModelCheckPoint, psnr




class Trainer(object):

    def __init__(self, model, criterion, optimizer, device='cpu', callbacks=None):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.callbacks = callbacks

    def train(self, epochs, train_dataloader, val_dataloader):

        train_len = len(train_dataloader)
        val_len = len(val_dataloader)
        print(f'train_step: {train_len}, val_step: {val_len}')

        for epoch in range(epochs):
            dt_now = datetime.datetime.now()
            print(dt_now)
            print('Train')
            self.model.train()
            epoch_time = time()
            train_loss = []
            val_loss = []
            for i, (inputs, labels) in enumerate(train_dataloader):
                step_time = time()
                inputs, labels = self.__trans_data(inputs, labels)
                loss = self.__step(inputs, labels)
                train_loss.append(loss.item())
                psnr_show = psnr(loss)
                if self.device is 'cuda':
                    print(f'\rEpoch: {epoch + 1:05d} / {epochs:05d}, Step: {i + 1:05d} / {train_len:05d}, Loss: {loss:.7f}, psnr: {psnr_show:.7f}, StepTime: {time() - step_time:.5f}sec, EpochTime: {time() - epoch_time:.5f}sec, Cache: {torch.cuda.memory_cached(0)/1024**3:.5f}GB', end='')
                elif self.device is 'cpu':
                    print(f'\rEpoch: {epoch + 1:05d} / {epochs:05d}, Step: {i + 1:05d} / {train_len:05d}, Loss: {loss:.7f}, psnr: {psnr_show:.7f},StepTime: {time() - step_time:.5f}sec, EpochTime: {time() - epoch_time:.5f}sec', end='')
                torch.cuda.empty_cache()
            print()
            print('Validation')
            self.model.eval()
            for i, (inputs, labels) in enumerate(val_dataloader):
                step_time = time()
                inputs, labels = self.__trans_data(inputs, labels)
                with torch.no_grad():
                    loss = self.__step(inputs, labels, train=False)
                val_loss.append(loss.item())
                psnr_show = psnr(loss)
                if self.device is 'cuda':
                    print(f'\rEpoch: {epoch + 1:05d} / {epochs:05d}, Step: {i + 1:05d} / {val_len:05d}, Loss: {loss:.7f}, psnr: {psnr_show:.7f}, StepTime: {time() - step_time:.5f}sec, EpochTime: {time() - epoch_time:.5f}sec, Cache: {torch.cuda.memory_cached(0)/1024**3:.5f}GB', end='')
                elif self.device is 'cpu':
                    print(f'\rEpoch: {epoch + 1:05d} / {epochs:05d}, Step: {i + 1:05d} / {val_len:05d}, Loss: {loss:.7f}, psnr: {psnr_show:.7f},StepTime: {time() - step_time:.5f}sec, EpochTime: {time() - epoch_time:.5f}sec', end='')
                torch.cuda.empty_cache()
            print()
            train_loss = np.mean(train_loss)
            val_loss = np.mean(val_loss)
            print('train_loss:', type(train_loss))
            if self.callbacks:
                for callback in self.callbacks:
                    callback.callback(self.model, epoch, loss=train_loss, val_loss=val_loss, save=True, device=self.device)
            print()

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


if __name__ == '__main__':

    resize = 256
    crop = 96
    batch_size = 2
    epochs = 5

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
    train_dataset = AutoColorDataset(train_path, train_list[:batch_size], transform)
    test_dataset = AutoColorDataset(test_path, test_list[:batch_size], transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(len(train_dataloader))
    print(len(test_dataloader))

    model = CAE([64, 128, 256]).to(device)
    summary(model, (1, 96, 96))
    criterion = torch.nn.MSELoss().to(device)
    param = list(model.parameters())
    optimizer = torch.optim.Adam(lr=1e-3, params=param)
    draw_cb = Draw_Output(test_path, test_list[:9], nrow=3, save_path='output', verbose=False)
    ckpt_cb = ModelCheckPoint('ckpt', 'CAE')

    trainer = Trainer_test(model, criterion, optimizer, device=device, callbacks=[draw_cb, ckpt_cb])
    trainer.train(epochs, train_dataloader, test_dataloader)
