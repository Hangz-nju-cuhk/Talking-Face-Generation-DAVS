from __future__ import print_function, division
import torch
import torch.nn as nn


class mfcc_encoder(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(mfcc_encoder, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3),
                             stride=(3, 2), padding=(1, 2), bias=use_bias)
        self.pool1 = nn.AvgPool2d((2, 2), 2)
        self.bn1 = norm_layer(64)
        self.conv2 = nn.Conv2d(64, 128, (3, 3), 2, 1, bias=use_bias)
        self.pool2 = nn.AvgPool2d(2,2)
        self.bn2 = norm_layer(128)
        self.conv3 = nn.Conv2d(128, 256, (3, 3), 1, 0, bias=use_bias)
        self.bn3 = norm_layer(256)
        self.conv4 = nn.Conv2d(256, 512, (2, 2), 1, bias=use_bias)

        self.bn5 = norm_layer(512)
        self.tanh = nn.Tanh()

    def forward(self, x):
        net1 = self.conv1(x)
        net1 = self.bn1(net1)
        net1 = self.relu(net1)

        net = self.conv2(net1)
        net = self.bn2(net)
        net = self.relu(net)

        net = self.conv3(net)
        net = self.bn3(net)
        net = self.relu(net)

        net = self.conv4(net)
        return net


class mfcc_encoder_alter(nn.Module):
    def __init__(self):
        super(mfcc_encoder_alter, self).__init__()
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 12), stride=(1,1), padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool2d(1, 3)
        self.conv2 = nn.Conv2d(64, 256, (3, 1), 1, (1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(1, 2)
        self.conv3 = nn.Conv2d(256, 512, (3, 1), 1, bias=False)

    def forward(self, x):
        net = self.conv1(x)
        net = self.relu(self.bn1(net))
        net = self.pool1(net)
        net = self.conv2(net)
        net = self.relu(self.bn2(net))
        net = self.pool2(net)
        net = self.conv3(net)
        return net


class mfcc_encoder_two(nn.Module):
    def __init__(self, opt):
        super(mfcc_encoder_two, self).__init__()
        self.opt = opt
        self.model1 = mfcc_encoder()
        self.model2 = mfcc_encoder_alter()
        self.fc = nn.Linear(1024, 256)

    def _forward(self, x):
        net1 = self.model1.forward(x)
        net2 = self.model2.forward(x)
        net = torch.cat((net1, net2), 1)
        net = net.view(-1, 1024)
        net = self.fc(net)
        return net

    def forward(self, x):
        x0 = x.view(-1, 1, self.opt.mfcc_length, self.opt.mfcc_width)
        net = self._forward(x0)
        net = net.view(x.size(0), -1, 256)
        return net
