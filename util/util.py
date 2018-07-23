from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import torch.nn as nn
import os
from Options_all import BaseOptions
import collections
config = BaseOptions().parse()


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    PIL_image = image_numpy

    return PIL_image.astype(imtype)

def tensor2image(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    PIL_image = image_numpy

    return PIL_image.astype(imtype)

def tensor2mfcc(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    PIL_image = image_numpy
    return PIL_image.astype(imtype)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_checkpoint(state, epoch, filename=config.name + '_checkpoint.pth.tar', step=0):
    torch.save(state, os.path.join(config.checkpoints_dir, str(epoch) + "_" + str(step) + "_" + filename))


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model


def load_checkpoint(resume_path, Model):
    resume_path = resume_path
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        total_steps = checkpoint['step']
        epoch = checkpoint['epoch']
        Model.ID_encoder = copy_state_dict(checkpoint['ID_encoder'], Model.ID_encoder)
        Model.Decoder = copy_state_dict(checkpoint['Decoder'], Model.Decoder)
        Model.mfcc_encoder = copy_state_dict(checkpoint['mfcc_encoder'], Model.mfcc_encoder)
        Model.lip_feature_encoder = copy_state_dict(checkpoint['lip_feature_encoder'], Model.lip_feature_encoder)
        Model.netD = copy_state_dict(checkpoint['netD'], Model.netD)
        Model.netD_mul = copy_state_dict(checkpoint['netD_mul'], Model.netD_mul)
        Model.ID_lip_discriminator = copy_state_dict(checkpoint['ID_lip_discriminator'], Model.ID_lip_discriminator)
        Model.model_fusion = copy_state_dict(checkpoint['model_fusion'], Model.model_fusion)
        Model.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        Model.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        print("=> loaded checkpoint '{}' (step {})"
              .format(resume_path, checkpoint['step']))
        return Model, total_steps, epoch
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))



def load_separately(opt, Model):
    print("=> loading checkpoint '{}'".format(opt.id_pretrain_path))
    id_pretrain = torch.load(opt.id_pretrain_path)
    Model.ID_encoder = copy_state_dict(id_pretrain['model_fusion'], Model.ID_encoder)
    print("=> loading checkpoint '{}'".format(opt.feature_extractor_path))
    feature_extractor_check = torch.load(opt.feature_extractor_path)
    Model.lip_feature_encoder = copy_state_dict(feature_extractor_check['face_encoder'], Model.lip_feature_encoder)
    Model.mfcc_encoder = copy_state_dict(feature_extractor_check['mfcc_encoder'], Model.mfcc_encoder)
    Model.model_fusion = copy_state_dict(feature_extractor_check['face_fusion'], Model.model_fusion)
    return Model


def load_test_checkpoint(resume_path, Model):
    resume_path = resume_path
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        total_steps = checkpoint['step']
        epoch = checkpoint['epoch']
        Model.ID_encoder = copy_state_dict(checkpoint['ID_encoder'], Model.ID_encoder)
        Model.Decoder = copy_state_dict(checkpoint['Decoder'], Model.Decoder)
        Model.mfcc_encoder = copy_state_dict(checkpoint['mfcc_encoder'], Model.mfcc_encoder)
        Model.lip_feature_encoder = copy_state_dict(checkpoint['lip_feature_encoder'], Model.lip_feature_encoder)
        # Model.model_fusion = copy_state_dict(checkpoint['model_fusion'], Model.model_fusion)

        print("=> loaded checkpoint '{}' (step {})"
              .format(resume_path, checkpoint['step']))
        return Model, total_steps, epoch
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))
