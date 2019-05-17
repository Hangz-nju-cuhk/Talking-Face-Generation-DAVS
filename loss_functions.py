from __future__ import print_function, division
import torch
import torch.nn as nn
import embedding_utils
from torch.autograd import Variable
import random
import Options

opt = Options.Config()

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, softlabel=False):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.softlabel = softlabel
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if self.softlabel:
            soft = random.random() * 0.1
        else:
            soft = 0
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label - soft)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label + soft)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

        self.sim = embedding_utils.sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]

        return cost_s.sum()


class SumLogSoftmaxLoss(nn.Module):
    def __init__(self, opt=opt):
        super(SumLogSoftmaxLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        out = self.logsoftmax(x)
        loss = - torch.mean(out)
        return loss
class L2SoftmaxLoss(nn.Module):
    def __init__(self, opt=opt):
        super(L2SoftmaxLoss, self).__init__()
        self.softmax = nn.Softmax()
        self.L2loss = nn.MSELoss()
        self.label = None

    def forward(self, x):
        out = self.softmax(x)
        self.label = Variable(torch.ones(out.size()).float() * (1 / x.size(1)), requires_grad=False).cuda()
        loss = self.L2loss(out, self.label)
        return loss

class L2ContrastiveLoss(nn.Module):
    """
    Compute L2 contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(L2ContrastiveLoss, self).__init__()
        self.margin = margin

        self.sim = embedding_utils.l2_sim

        self.max_violation = max_violation

    def forward(self, feature1, feature2):
        # compute image-sentence score matrix
        scores = self.sim(feature1, feature2)
        # diagonal = scores.diag().view(feature1.size(0), 1)
        diagonal_dist = scores.diag()
        # d1 = diagonal.expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin - scores).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]

        loss = (torch.sum(cost_s ** 2) + torch.sum(diagonal_dist ** 2)) / (2 * feature1.size(0))

        return loss
