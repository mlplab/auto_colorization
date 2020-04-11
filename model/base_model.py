# coding: utf-8


import torch


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
        layers = [torch.nn.ConvTransBlockpose2d(in_feature, out_feature, kernel_size=2, stride=2)]
        features = [out_feature for _ in range(num_layer)]
        for i in range(len(features) - 1):
            layers.append(torch.nn.Conv2d(features[i], features[i + 1], kernel_size=kernel, padding=1))
            layers.append(torch.nn.BatchNorm2d(features[i + 1]))
            layers.append(torch.nn.ReLU())
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


class Conv_Block(torch.nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size, stride, padding, norm=True):
        super(Conv_Block, self).__init__()
        layer = []
        if norm is True:
            layer.append(torch.nn.BatchNorm2d(input_ch))
        layer.append(torch.nn.ReLU())
        #                             kernel_size=kernel_size, stride=stride,
        #                             padding=padding)
        layer.append(torch.nn.Conv2d(input_ch, output_ch,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding))
        self.layer = torch.nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)


class Bottoleneck(torch.nn.Module):

    def __init__(self, input_ch, k):
        super(Bottoleneck, self).__init__()
        bottoleneck = []
        bottoleneck.append(Conv_Block(input_ch, 128, 1, 1, 0))
        bottoleneck.append(Conv_Block(128, k, 3, 1, 1))
        self.bottoleneck = torch.nn.Sequential(*bottoleneck)

    def forward(self, x):
        return self.bottoleneck(x)


class DenseBlock(torch.nn.Module):

    def __init__(self, input_ch, k, layer_num):
        super(DenseBlock, self).__init__()
        bottoleneck = []
        for i in range(layer_num):
            bottoleneck.append(Bottoleneck(input_ch, k))
            input_ch += k
        self.bottoleneck = torch.nn.Sequential(*bottoleneck)

    def forward(self, x):
        for i, bottoleneck in enumerate(self.bottoleneck):
            growth = bottoleneck(x)
            x = torch.cat((x, growth), dim=1)
        return x


class TransBlock(torch.nn.Module):

    def __init__(self, input_ch, compress=.5):
        super(TransBlock, self).__init__()
        self.conv1_1 = Conv_Block(input_ch, int(input_ch * compress), 1, 1, 0, norm=False)
        self.ave_pool = torch.nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.ave_pool(x)
        return x
