from __future__ import print_function, division
import torch
import torch.nn as nn
import functools

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        # m.weight.data.fill_(1)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class ModelFusion(nn.Module):
    def __init__(self, config):
        super(ModelFusion, self).__init__()
        self.fc_1 = nn.Linear(config.pred_length * 256, 512)
        self.fc_2 = nn.Linear(512, config.label_size)
        self.relu = nn.ReLU(True)
        self.sig = nn.Sigmoid()
        self.config = config
        self.dis = nn.Linear(512, 1)
        if not config.resume:
            self.fc_1.weight.data.normal_(0, 0.0001)
            self.fc_1.bias.data.zero_()

    def forward(self, x):
        x = x.view(-1, self.config.pred_length * 256)
        net = self.fc_1(x)
        net0 = self.relu(net)
        net = self.fc_2(net0)
        # # net0 = self.dropout(net)
        #
        # dis_feature = self.sig(self.dis(net0))
        return net


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
        dis1 = self.sig(net)
        return dis1
