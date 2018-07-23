from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
import networks


class VggEncoder(nn.Module):
    def __init__(self, config):
        super(VggEncoder, self).__init__()
        self.config = config
        self.relu = nn.ReLU(True)

        self.conv1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)

        self.conv2 = nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)

        self.conv3 = nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1))
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))

        self.bn4 = nn.BatchNorm2d(256)

        self.pool3 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        self.conv5 = nn.Conv2d(256, 512, (3, 3), (1, 1), 1)

        self.bn5 = nn.BatchNorm2d(512)

        self.pool4 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        self.conv6 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, (3, 3), (1, 1), 1)
        self.bn7 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)

        self.pool6 = nn.AvgPool2d((8, 8))

        self.fc = nn.Linear(512, 500)


    def forward(self, x):
        x = x.view(-1, self.config.image_channel_size, self.config.image_size, self.config.image_size)
        net1 = self.conv1(x)
        net1 = self.bn1(net1)
        net1 = self.relu(net1)

        net2 = self.pool1(net1)    # shape 128
        net2 = self.conv2(net2)
        net2 = self.relu(self.bn2(net2))

        net3 = self.pool2(net2)   # shape 64


        net3 = self.conv3(net3)
        net3 = self.bn3(net3)
        net3 = self.relu(net3)

        net4 = self.conv4(net3)
        net4 = self.bn4(net4)
        net4 = self.relu(net4)

        net5 = self.pool3(net4)   # shape 32

        net5 = self.conv5(net5)
        net5 = self.bn5(net5)
        net5 = self.relu(net5)

        net6 = self.pool4(net5)  # shape 16

        net6 = self.conv6(net6)
        net6 = self.bn6(net6)
        net6 = self.relu(net6)

        net7 = self.conv7(net6)
        net7 = self.bn7(net7)
        net7 = self.relu(net7)

        net8 = self.pool5(net7)
        net9 = self.pool6(net8)  # shape 8
        net9 = net9.view(-1, 512)
        net0 = self.fc(net9)
        net = [net9, net8,  net7, net5, net4, net2, net0]
        return net


class VggDecoder(nn.Module):
    def __init__(self, config, ngf=64, norm_layer=nn.BatchNorm2d):
        super(VggDecoder, self).__init__()
        self.config = config
        use_bias = norm_layer == nn.InstanceNorm2d
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.8)
        self.avgpool = nn.AvgPool2d(7, 7)
        self.uprelu = nn.ReLU(True)

        self.upconv00 = nn.ConvTranspose2d(512 + 256, 512,
                                    kernel_size=8, stride=1,
                                    padding=0, bias=use_bias)
        self.upnorm00 = norm_layer(512)
        self.upconv0 = nn.ConvTranspose2d(512, 512,
                                    kernel_size=4, stride=2,
                                    padding=1, bias=use_bias)
        self.upnorm0 = norm_layer(512)
        self.upconv1 = nn.ConvTranspose2d(512, 512,
                                    kernel_size=4, stride=2,
                                    padding=1, bias=use_bias)
        self.upnorm1 = norm_layer(512)
        self.upconv2 = nn.ConvTranspose2d(512 * 2, 256,
                                    kernel_size=4, stride=2,
                                    padding=1)
        self.upnorm2 = norm_layer(256)
        self.upconv3 = nn.ConvTranspose2d(256, 128,
                                    kernel_size=4, stride=2,
                                    padding=1, bias=use_bias)
        self.upnorm3 = norm_layer(128)
        self.upconv4 = nn.ConvTranspose2d(128, 64,
                                    kernel_size=4, stride=2,
                                    padding=1, bias=use_bias)
        self.upnorm4 = norm_layer(64)
        self.upconv5 = nn.ConvTranspose2d(64, ngf,
                                    kernel_size=3, stride=1,
                                    padding=1, bias=use_bias)
        self.upnorm5 = norm_layer(ngf)
        self.upconv6 = nn.Conv2d(64, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.apply(networks.weights_init)

    def forward(self, id_feature, mouth_feature):
        id_feature0 = id_feature[0].view(-1, 512, 1, 1)
        mouth_feature = mouth_feature.view(-1, self.config.feature_length, 1, 1)
        whole_feature = torch.cat((id_feature0, mouth_feature), dim=1)
        net = self.upconv00(whole_feature)
        net = self.upnorm00(net)
        for i in range(0, 5):
            if i == 2:
                net = torch.cat((id_feature[i + 1], net), 1)
            net = self._modules['upconv' + str(i)](net)
            net = self._modules['upnorm' + str(i)](net)
            net = self.uprelu(net)
        net = self.upconv5(net)
        net = self.upnorm5(net)
        net = self.uprelu(net)
        net = self.upconv6(net)

        net = self.tanh(net)
        net = (net + 1) / 2.0
        return net