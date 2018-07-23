from __future__ import print_function, division
import torch
import torch.nn as nn
from Options_all import BaseOptions
opt = BaseOptions().parse()


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, name, conv_std=0.058925565,
                 kernel_size=3, padding=1, stride=1):
        super(BasicBlock, self).__init__()
        if isinstance(name, str):
            self.name = name
        else:
            raise Exception("name should be str")
        self.conv_std = conv_std
        self.add_module('conv' + self.name + "_a", nn.Conv2d(inplanes, outplanes, padding=padding, kernel_size=kernel_size, stride=stride))
        self.add_module('conv' + self.name + "_b",  nn.Conv2d(inplanes, outplanes, padding=padding, kernel_size=kernel_size, stride=stride))
        self.add_module('conv' + self.name + "_a_bn", nn.BatchNorm2d(outplanes))
        self.add_module('conv' + self.name + "_b_bn", nn.BatchNorm2d(outplanes))
        self.initial()

    def forward(self, x):
        data_a = self._modules['conv' + self.name + "_a"](x)
        data_b = self._modules['conv' + self.name + "_b"](x)
        data_a = self._modules['conv' + self.name + "_a_bn"](data_a)
        data_b = self._modules['conv' + self.name + "_b_bn"](data_b)
        data_output = torch.max(data_a, data_b)
        return data_output

    def initial(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight, std=self.conv_std)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)


class IdentityEncoder(nn.Module):
    def __init__(self, opt=opt):
        super(IdentityEncoder, self).__init__()
        self.opt = opt
        self.add_module('block' + str(01), BasicBlock(3, 32, name="01", conv_std=0.025253814, kernel_size=7, stride=2, padding=3))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.add_module('block' + str(11), BasicBlock(32, 32, name="11", conv_std=0.058925565))
        self.add_module('block' + str(12), BasicBlock(32, 32, name="12", conv_std=0.058925565))
        self.add_module('block' + str(13), BasicBlock(32, 64, name="13", conv_std=0.041666668))
        self.add_module('block' + str(14), BasicBlock(64, 64, name="14", conv_std=0.041666668))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.add_module('block' + str(21), BasicBlock(64, 96, name="21", conv_std=0.034020692))
        self.add_module('block' + str(22), BasicBlock(96, 96, name="22", conv_std=0.034020692))
        self.add_module('block' + str(23), BasicBlock(96, 128, name="23", conv_std=0.029462783))
        self.add_module('block' + str(24), BasicBlock(128, 128, name="24", conv_std=0.029462783))
        self.add_module('block' + str(25), BasicBlock(128, 160, name="25", conv_std=0.026352314))
        self.add_module('block' + str(26), BasicBlock(160, 160, name="26", conv_std=0.026352314))
        self.pool3 = nn.MaxPool2d(2, 2)
        self.add_module('block' + str(31), BasicBlock(160, 320, name="31", conv_std=0.0186339))
        self.add_module('block' + str(32), BasicBlock(320, 320, name="32", conv_std=0.0186339))
        self.add_module('block' + str(33), BasicBlock(320, 320, name="33", conv_std=0.0186339))
        self.add_module('block' + str(34), BasicBlock(320, 320, name="34", conv_std=0.0186339))
        self.pool4 = nn.MaxPool2d(2, 2)
        self.add_module('block' + str(41), BasicBlock(320, 256, name="41", conv_std=0.020833334, kernel_size=5, padding=2))
        self.add_module('block' + str(42), BasicBlock(256, 256, name="42", conv_std=0.020833334, kernel_size=5, padding=2))
        self.dropout = nn.Dropout2d(0.5)
        self.avgpool = nn.AvgPool2d((8, 8), stride=1)

    def forward(self, x):
        x = x.view(-1, self.opt.image_channel_size, self.opt.image_size, self.opt.image_size)
        net0 = self._modules['block' + str(01)](x)
        net1 = self.pool1(net0)
        for j1 in range(11, 15):
            net1 = self._modules['block' + str(j1)](net1)
        net2 = self.pool2(net1)
        for j2 in range(21, 27):
            net2 = self._modules['block' + str(j2)](net2)
        net3 = self.pool3(net2)
        for j3 in range(31, 35):
            net3 = self._modules['block' + str(j3)](net3)
        net4 = self.pool4(net3)
        for j4 in range(41, 43):
            net4 = self._modules['block' + str(j4)](net4)

        net = self.avgpool(net4)
        net = net.view(-1, 256)

        return [net, net4, net3, net2, net1]



