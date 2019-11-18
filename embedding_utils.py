from __future__ import print_function, division
import torch
import numpy as np
import torch.nn as nn
import os
import shutil
from sklearn.metrics.pairwise import euclidean_distances
import torch.nn.functional as F
from Options import Config
config = Config().parse()


def to_np(x):
    return x.data.cpu().numpy()


def save_checkpoint(state, epoch, is_best, filename=config.name + '_checkpoint.pth.tar'):
    if not os.path.exists(config.checkpoints_dir):
        os.mkdir(config.checkpoints_dir)
    torch.save(state, os.path.join(config.checkpoints_dir, str(epoch) + "_" + filename))
    if is_best:
        shutil.copyfile(os.path.join(config.checkpoints_dir, str(epoch) + "_" + filename), config.name + '_model_best.pth.tar')


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


def load_checkpoint(resume_path, model):

    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.start_step = checkpoint['step']
        epoch = checkpoint['epoch']
        model.best_acc = checkpoint['best_acc']
        model.min_loss = checkpoint['min_loss']
        model.mfcc_encoder = copy_state_dict(checkpoint['mfcc_encoder'], model.mfcc_encoder)
        model.model_fusion = copy_state_dict(checkpoint['model_fusion'], model.model_fusion)
        model.face_encoder = copy_state_dict(checkpoint['face_encoder'], model.face_encoder)
        model.face_fusion = copy_state_dict(checkpoint['face_fusion'], model.face_fusion)
        model.discriminator_audio = copy_state_dict(checkpoint['discriminator_audio'], model.discriminator_audio)
        # model.discriminator_image = copy_state_dict(checkpoint['discriminator_image'], model.discriminator_image)
        model.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        model.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        print("=> loaded checkpoint '{}' (step {})"
              .format(resume_path, checkpoint['step']))
        return model, epoch
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))


def load_ini(resume_path1, resume_path2, model):
    print("=> loading checkpoint '{}'".format(resume_path1))
    checkpoint1 = torch.load(resume_path1)
    print("=> loading checkpoint '{}'".format(resume_path2))
    checkpoint2 = torch.load(resume_path2)
    model.mfcc_encoder = copy_state_dict(checkpoint1['image_model'], model.mfcc_encoder)
    model.model_fusion = copy_state_dict(checkpoint2['mfcc_fusion'], model.model_fusion)
    return model


def adjust_learning_rate(audio_model, config, loss):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if loss < config.loss_buffer:
        # config.lr1 = config.lr1 + 1e-8
        pass
    else:
        config.lr = config.lr * 0.5
    config.loss_buffer = loss

    for param_group in audio_model.optimizer.param_groups:
        param_group['lr'] = config.lr


def load_synthesis_checkpoint(resume_path, model):

    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.mfcc_encoder = copy_state_dict(checkpoint['mfcc_encoder'], model.mfcc_encoder)
        return model
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))

def l2_sim(feature1, feature2):
    Feature = feature1.expand(feature1.size(0), feature1.size(0), feature1.size(1)).transpose(0, 1)
    return torch.norm(Feature - feature2, p=2, dim=2)


def l2_norm(x):
    x_norm = F.normalize(x, p=2, dim=1)
    return x_norm


def sim(feature1, feature2):
    """Cosine similarity between all the image and sentence pairs
    """
    return feature1.mm(feature2.t())


def sentence_to_video(clips_embed, captions_embed, return_ranks = False):
    captions_num = captions_embed.shape[0]
    #index_list = []
    ranks = np.zeros(captions_num)
    top1 = np.zeros(captions_num)

    for i in range(captions_num):
        # caption dim : 1 * embed_size; clips_embed dim: num * embed_size
        # d : 1 * num : represent the similarity between this caption and each clip
        caption = captions_embed[i]
        d = np.dot(caption, clips_embed.T).flatten()
        inds = np.argsort(d)[::-1]

        rank = np.where(inds == i)[0][0]
        ranks[i] = rank
        top1[i] = inds[0]

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    # r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    #plus 1 because the index starts from 0
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, r50, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, r50, medr, meanr)


def L2retrieval(clips_embed, captions_embed, return_ranks = False):
    captions_num = captions_embed.shape[0]
    #index_list = []
    ranks = np.zeros(captions_num)
    top1 = np.zeros(captions_num)
    import time
    t1 = time.time()
    d = euclidean_distances(captions_embed, clips_embed)
    inds = np.argsort(d)
    num = np.arange(captions_num).reshape(captions_num, 1)
    ranks = np.where(inds == num)[1]
    top1 = inds[:, 0]
    t2 = time.time()
    print((t2 - t1))
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    # r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    #plus 1 because the index starts from 0
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, r50, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, r50, medr, meanr)
