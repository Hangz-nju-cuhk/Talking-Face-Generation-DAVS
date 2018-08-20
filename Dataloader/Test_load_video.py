from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2


def Test_Outside_Loader(path, A_path, config, require_video=True):
    loader = {}
    video_data_length = config.test_audio_video_length
    video_pair = range(2, video_data_length)
    im_pth = []

    video_block = np.zeros((config.test_audio_video_length,
                            config.image_size,
                            config.image_size,
                            config.image_channel_size))

    crop_x = 2
    crop_y = 2
    A_image = cv2.cvtColor(cv2.imread(A_path), cv2.COLOR_BGR2RGB)
    A_image = A_image.astype(np.float)
    A_image = A_image / 255
    A_image = cv2.resize(A_image[crop_x:crop_x + config.image_size, crop_y:crop_y + config.image_size],
                         (config.image_size, config.image_size))
    if os.path.isdir(path):

        k1 = 0

        for image_num in video_pair:

            image_path = os.path.join(path, str(image_num) + '.jpg')
            im_pth.append(image_path)
            if os.path.exists(image_path):
                image = cv2.imread(image_path)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = image / 255

                video_block[k1] = image[crop_x:crop_x + config.image_size, crop_y:crop_y + config.image_size]
            else:
                print("video_block = 0")
                break

            k1 += 1

        video_block = video_block.transpose((0, 3, 1, 2))
        A_image = A_image.transpose((2, 0, 1))
        loader['A'] = A_image
        if require_video:
            loader['B'] = video_block
        loader['A_path'] = A_path
        loader['B_path'] = im_pth
    return loader


class Test_VideoFolder(Dataset):

    def __init__(self, root, A_path, config, transform=None, target_transform=None,
                 loader=Test_Outside_Loader, mode='test'):

        self.root = root
        self.A_path = A_path
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.config = config
        self.mode = mode
        self.vid = self.loader(self.root, self.A_path,  config=self.config)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        loader = {}

        loader['A'] = self.vid['A']
        loader['B'] = self.vid['B'][index:index + self.config.sequence_length, :, :, :]
        loader['A_path'] = self.A_path
        loader['B_path'] = self.vid['B_path'][index:self.config.sequence_length + index]
        return loader

    def __len__(self):
        return self.config.test_audio_video_length - self.config.sequence_length + 1
