# coding: utf-8


import os
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision




def psnr(loss):
    return 10 * torch.log10(1. ** 2 / loss)


class Draw_Output(object):
    
    def __init__(self, img_path, output_data, save_path='output', verbose=False, nrow=8):
        '''
        Parameters
        ---
        img_path: str
            image dataset path
        output_data: list
            draw output data path
        save_path: str (default: 'output')
            output img path
        verbose: bool (default: False)
            verbose
        '''
        self.img_path = img_path
        self.output_data = output_data
        self.data_num = len(output_data)
        self.save_path = save_path
        self.verbose = verbose
        self.input_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor()
        ])
        self.output_transform = torchvision.transforms.ToPILImage()
        self.nrow = nrow

        ###########################################################
        # Make output directory
        ###########################################################        
        if os.path.exists(save_path) is True:
            shutil.rmtree(save_path)
        os.mkdir(save_path)
        if os.path.exists(save_path + '/all_imgs') is True:
            shutil.rmtree(save_path + '/all_imgs')
        os.mkdir(save_path + '/all_imgs')
        ###########################################################
        # Draw Label Img
        ###########################################################
        labels = []
        for data in self.output_data:
            label = self.input_transform(Image.open(os.path.join(self.img_path, data)).convert('RGB'))
            labels.append(label)
        self.labels = torch.cat(labels).reshape(len(labels), *labels[0].shape)
        labels_np = torchvision.utils.make_grid(self.labels, nrow=nrow, padding=10)
        labels_np = labels_np.numpy()
        self.labels_np = np.transpose(labels_np, (1, 2, 0))
        del labels, labels_np
        torchvision.utils.save_image(self.labels, os.path.join(save_path, f'labels.jpg'), nrow=nrow, padding=10)
        
    
    def callback(self, model, epoch, *args, **kwargs):
        if 'save' not in kwargs.keys():
            assert 'None save mode'
        else:
            save = kwargs['save']
        device = kwargs['device']
        self.epoch_save_path = os.path.join(self.save_path, f'epoch{epoch}')
        output_imgs = []
        model.eval()
        for i, data in enumerate(self.output_data):
            img = self.input_transform(Image.open(os.path.join(self.img_path, data)).convert('L')).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img).squeeze().to('cpu')
            output_imgs.append(output)
        output_imgs = torch.cat(output_imgs).reshape(len(output_imgs), *output_imgs[0].shape)
        if self.verbose is True:
            self.__show_output_img_list(output_imgs)
            # self.__show_output_img_list(self.labels)
        if save is True:
            torchvision.utils.save_image(output_imgs, os.path.join(self.save_path, f'all_imgs/all_imgs_{epoch}.jpg'), nrow=self.nrow, padding=10)
        del output_imgs
        return self

    def __draw_output_label(self, output, label, data):
        output = torch.cat((label, output), dim=2)
        output = self.output_transform(output)
        output.save(os.path.join(self.epoch_save_path, data))
        if self.verbose is True:
            print(f'\rDraw Output {data}', end='')
        return self

    def __show_output_img_list(self, output_imgs):
        plt.figure(figsize=(16, 9))
        output_imgs_np = torchvision.utils.make_grid(output_imgs, nrow=self.nrow, padding=10)
        output_imgs_np = output_imgs_np.numpy()
        output_imgs_np = np.transpose(output_imgs_np, (1, 2, 0))
        plt.subplot(1, 2, 1)
        plot_img(output_imgs_np, 'Predict')
        plt.subplot(1, 2, 2)
        plot_img(self.labels_np, 'Label')
        plt.show()
        del output_imgs_np
        return self

    @staticmethod
    def plot_img(output_imgs, title):
        plt.imshow(output_imgs)
        plt.title('Predict')
        plt.xticks(color="None")
        plt.yticks(color="None")
        plt.tick_params(length=0)
        return self


class ModelCheckPoint(object):

    def __init__(self, checkpoint_path, model_name, partience=1, verbose=True):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.partience = partience
        self.verbose = verbose
        if os.path.exists(self.checkpoint_path):
            shutil.rmtree(self.checkpoint_path)
        os.makedirs(self.checkpoint_path)

    def callback(self, model, epoch, *args, **kwargs):
        if 'loss' not in kwargs.keys() and 'val_loss' not in kwargs.keys():
            assert 'None Loss'
        else:
            loss = kwargs['loss']
            val_loss = kwargs['val_loss']
        checkpoint_name = os.path.join(self.checkpoint_path, self.model_name + f'_epoch_{epoch:05d}_loss_{loss:.5f}_valloss_{val_loss:.5f}.pth')
        if epoch % self.partience == 0:
            torch.save(model.state_dict(), checkpoint_name)
            if self.verbose is True:
                print(f'CheckPoint Saved by {checkpoint_name}')
        return self
