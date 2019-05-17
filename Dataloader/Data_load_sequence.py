from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import Options
import cv2
import shutil
config = Options.Config()


def find_classes(dir, config=config):
    classes = [str(d) for d in range(config.label_size)]
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, mode, config=config):
    videos = []
    dir = os.path.expanduser(dir)
    classes = [str(d) for d in range(config.label_size)]
    for target in classes:
        d = os.path.join(dir, target)

        if not os.path.isdir(d):
            continue
        listd = sorted(os.listdir(d))
        #if mode == 'val':
            #listd = random.sample(listd, 10)
        for fnames in listd:
            path = os.path.join(d, fnames)
            if os.path.isdir(os.path.join(d, fnames)):
                if os.path.exists(path):
                    item = (path, class_to_idx[target])
                    videos.append(item)
    return videos


def lip_reading_loader(path, config=config, mode='train', random_crop=True,
                       ini='fan'):
    loader = {}
    pair = np.arange(2, 27)
    im_pth = []
    video_block = np.zeros((25,
                            config.image_size,
                            config.image_size,
                            config.image_channel_size))
    mfcc_block = np.zeros(( 25, 1,
                            config.mfcc_length,
                            config.mfcc_width,
                            ))

    if os.path.isdir(path):
        for block in (os.listdir(path)):
            block_dir = os.path.join(path, block)
            crop_x = 2
            crop_y = 2
            if mode == 'train':
                flip = np.random.randint(0, 2)
                if random_crop:
                    crop_x = np.random.randint(0, 5)
                    crop_y = np.random.randint(0, 5)
            else:
                flip = 0

            if os.path.isdir(block_dir):
                if block == config.image_block_name:
                    k1 = 0
                    for image_num in pair:
                        image_path = os.path.join(block_dir, str(image_num) + '.jpg')
                        im_pth.append(image_path)
                        if os.path.exists(image_path):
                            image = cv2.imread(image_path)
                            if flip == 1:
                                image = np.fliplr(image)

                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            if ini == 'fan':
                                image = image / 255

                            video_block[k1] = image[crop_x:crop_x + config.image_size, crop_y:crop_y + config.image_size]
                        else:
                            print("video_block = 0")
                            shutil.rmtree(path)
                            break
                        k1 += 1

                if block == 'mfcc20':
                    if config.require_audio:
                        k4 = 0
                        for mfcc_num in pair:
                            # for s in range(-1,2):
                            mfcc_path = os.path.join(block_dir, str(mfcc_num) + '.bin')
                            if os.path.exists(mfcc_path):
                                mfcc = np.fromfile(mfcc_path)
                                mfcc = mfcc.reshape(20, 12)
                                mfcc_block[k4, 0, :, :] = mfcc

                                k4 += 1
                            else:
                                raise ("mfccs = 0")

        video_block = video_block.transpose((0, 3, 1, 2))
        loader['video'] = video_block
        loader['mfcc20'] = mfcc_block
        loader['A_path'] = im_pth[0]
        loader['B_path'] = im_pth[1:]
        # loader['label_map'] = label_map_block[:, 1:, :, :]
        if not np.abs(np.mean(mfcc_block)) < 1e5:
            print(np.mean(mfcc_block))
            print(im_pth)
            shutil.rmtree(path)
    return loader


class VideoFolder(Dataset):
    def __init__(self, root, config=config, transform=None, target_transform=None,
                 loader=lip_reading_loader, mode='train'):
        classes, class_to_idx = find_classes(root)
        videos = make_dataset(root, class_to_idx, mode)
        if len(videos) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"))
        self.root = root
        self.vids = videos
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.config = config
        self.mode = mode

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.vids[index]
        vid = self.loader(path, config=self.config, mode=self.mode)

        return vid, target

    def __len__(self):
        return len(self.vids)

