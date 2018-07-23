from __future__ import print_function, division
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, name, nums=3,
                 kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.nums = nums
        self.relu = nn.ReLU(True)
        if isinstance(name, str):
            self.name = name
        else:
            raise Exception("name should be str")
        for i in range(self.nums):
            self.add_module('conv' + self.name + "_" + str(i), nn.Conv2d(inplanes, outplanes, padding=padding, kernel_size=kernel_size, stride=stride))
            self.add_module('conv' + self.name + "_" + str(i) + "_bn", nn.BatchNorm2d(outplanes))
            inplanes = outplanes
        self.initial()

    def forward(self, x):
        net = x
        for i in range(self.nums):
            net = self._modules['conv' + self.name + "_" + str(i)](net)
            net = self._modules['conv' + self.name + "_" + str(i) + "_bn"](net)
            net = self.relu(net)
        return net

    def initial(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)


class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.opt = opt
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.deconv1_1_new = nn.ConvTranspose2d(512, 512, (4, 4), 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(512)
        self.convblock1 = ConvBlock(512, 512, "1", nums=2)
        self.convblock2 = ConvBlock(512, 512, "2", nums=3)
        self.convblock3 = ConvBlock(512, 256, "3", nums=4)
        self.convblock4 = ConvBlock(256 + 160, 256, "4", nums=4)
        self.convblock5 = ConvBlock(256, 128, "5", nums=3)
        self.convblock6 = ConvBlock(128, 64, "6", nums=2)
        self.conv7_1 = nn.ConvTranspose2d(64, 32, 3, 1, 1)
        self.conv7_1_bn = nn.BatchNorm2d(32)
        self.conv7_2 = nn.ConvTranspose2d(32, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, id_feature, mouth_feature):
        id_feature0 = id_feature[0].view(-1, self.opt.feature_length, 1, 1)
        mouth_feature = mouth_feature.view(-1, self.opt.feature_length, 1, 1)
        whole_feature = torch.cat((id_feature0, mouth_feature), dim=1)
        net = self.deconv1_1_new(whole_feature)
        net = self.relu(self.deconv1_1_bn(net))
        for i in range(6):
            if i == 3:
                net = torch.cat((id_feature[i], net), 1)
            net = self._modules['convblock' + str(i + 1)](net)
            net = self.upsample(net)
        net = self.conv7_1(net)
        net = self.relu(self.conv7_1_bn(net))
        net = self.conv7_2(net)
        net = self.tanh(net)
        net = (net + 1) / 2.0
        return net

