from __future__ import print_function, division
import torch
import torch.nn as nn
import functools
from Options import Config
config = Config().parse()


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(Discriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
        self.relu = nn.LeakyReLU(0.2, True)
        self.n_layers = n_layers
        self.use_sigmoid = use_sigmoid
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            self.add_module('conv2_' + str(n), nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias))
            self.add_module('norm_' + str(n), norm_layer(ndf * nf_mult))
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)

        self.conv3 = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias)
        self.norm3 = norm_layer(ndf * nf_mult)
        self.conv4 = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        self.mfcc_conv1 = nn.Conv2d(int(input_nc / 3), 64, kernel_size=(3, 3),
                             stride=(3, 2), padding=(1, 2), bias=use_bias)
        self.mfcc_bn1 = norm_layer(64)
        self.mfcc_conv2 = nn.Conv2d(64, 128, (3, 3), 2, 1, bias=use_bias)
        self.mfcc_bn2 = norm_layer(128)
        self.mfcc_conv3 = nn.Conv2d(128, 256, (3, 3), 1, 0, bias=use_bias)
        self.mfcc_bn3 = norm_layer(256)
        self.mfcc_conv4 = nn.Conv2d(256, 256, (2, 2), 1, bias=use_bias)
        self.mfcc_bn4 = norm_layer(256)
        self.conv6 = nn.Conv2d(512, 512, 3, 1, bias=use_bias)
        # self.conv7 = nn.Conv2d(512, 512, 1, 1, bias=use_bias)
        self.conv8 = nn.Conv2d(512, 1, 1, 1, bias=use_bias)
        self.bn6 = norm_layer(512)
        self.bn7 = norm_layer(512)
        self.bn8 = norm_layer(1)
        self.bn9 = norm_layer(512)
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv9 = nn.Conv2d(512, 512, 7, 1)
        self.fc = nn.Linear(256 + 512, 512)
        self.fc2 = nn.Linear(512, 1)
        if use_sigmoid:
            self.sig = nn.Sigmoid()

    def forward(self, input, audio):
        net = self.conv1(input)
        netn = self.relu(net)
        for n in range(1, self.n_layers):
            netn = self._modules['conv2_' + str(n)](netn)
            netn = self._modules['norm_' + str(n)](netn)
            netn = self.relu(netn)
        net = self.conv3(netn)
        net = self.norm3(net)
        net = self.relu(net)
        net2 = self.conv4(net)
        if self.use_sigmoid:
            net2 = self.sig(net2)
        net = self.conv6(net[:, :, 19:28, 11:20])
        net = self.bn6(net)
        net = self.relu(net)
        mfcc_encode = self._modules['mfcc_conv' + str(1)](audio)
        mfcc_encode = self._modules['mfcc_bn' + str(1)](mfcc_encode)
        for i in range(2, 5):
            mfcc_encode = self._modules['mfcc_conv' + str(i)](mfcc_encode)
            mfcc_encode = self._modules['mfcc_bn' + str(i)](mfcc_encode)

        net = self.conv9(net)
        net = self.relu(net)
        net = torch.cat((net, mfcc_encode), dim=1)
        net = net.view(-1, 256 + 512)
        net = self.fc(net)
        net = self.fc2(net)
        net = self.sig(net)
        return [net2, net]


class discriminator_audio(nn.Module):
    def __init__(self):
        super(discriminator_audio, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU(True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 256)
        net = self.fc1(x)
        net = self.fc2(self.relu(net))
        dis = self.sig(net)
        return dis


class ID_fc(nn.Module):
    def __init__(self, config=config):
        super(ID_fc, self).__init__()
        self.config = config
        self.fc_1 = nn.Linear(config.disfc_length * 256, 500)
        self.fc_2 = nn.Linear(512, config.label_size)
        self.relu = nn.ReLU(True)
        self.sig = nn.Sigmoid()
        self.dis = nn.Linear(512, 1)
        if not config.resume:
            self.fc_1.weight.data.normal_(0, 0.0001)
            self.fc_1.bias.data.zero_()

    def forward(self, x):
        x = x.view(-1, self.config.disfc_length * 256)
        net = self.fc_1(x)
        return net


class ID_dis32(nn.Module):
    def __init__(self, config=config, feature_length=64):
        super(ID_dis32, self).__init__()
        self.feature_length = feature_length
        self.conv6 = nn.Conv2d(self.feature_length, 1, 3, 2, 1)
        self.fc = nn.Linear(1024, 128)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.5)
        self.fc_2 = nn.Linear(128 * config.disfc_length, 500)
        if not config.resume:
            self.fc_2.weight.data.normal_(0, 0.0001)
            self.fc_2.bias.data.zero_()

    def _forward(self, x):
        net = self.conv6(x)
        net = self.relu(net)
        net = net.view(-1, 1024)

        net = self.fc(net)
        return net

    def forward(self, x, feature=False):
        x = x.view(-1, self.feature_length, 64, 64)
        net = self._forward(x)
        net0 = net.view(-1, self.config.disfc_length * 128)
        net = self.dropout(net0)
        net = self.fc_2(net)

        return net


class Face_ID_fc(nn.Module):
    def __init__(self, config=config):
        super(Face_ID_fc, self).__init__()
        self.fc = nn.Linear(256, config.id_label_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x0 = x.view(-1, 256)
        net = self.dropout(x0)
        net = self.fc(net)
        return net
