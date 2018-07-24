"""
The test model for generation based on video
"""

from __future__ import print_function, division

from collections import OrderedDict

import torch
from torch.autograd import Variable
import network.FAN_feature_extractor as FAN_feature_extractor
import network.IdentityEncoder as IdentityEncoder
import network.mfcc_networks as mfcc_networks
import network.Decoder_networks as Decoder_network
import util.util as util


class GenModel():

    def __init__(self, opt):
        self.opt = opt
        self.Tensor = torch.cuda.FloatTensor if opt.cuda_on else torch.Tensor
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.image_channel_size,
                                   opt.image_size, opt.image_size)
        self.input_B = self.Tensor(opt.batchSize, opt.test_audio_video_length, opt.image_channel_size,
                                   opt.image_size, opt.image_size)

        self.ID_encoder = IdentityEncoder.IdentityEncoder(opt)

        self.Decoder = Decoder_network.Decoder(opt)

        self.mfcc_encoder = mfcc_networks.mfcc_encoder_two(opt)

        self.lip_feature_encoder = FAN_feature_extractor.FanFusion(opt)

        self.criterionL1 = torch.nn.L1Loss()

        if torch.cuda.is_available():
            if opt.cuda_on:
                if opt.mul_gpu:
                    self.ID_encoder = torch.nn.DataParallel(self.ID_encoder)
                    self.Decoder = torch.nn.DataParallel(self.Decoder)
                    self.mfcc_encoder = torch.nn.DataParallel(self.mfcc_encoder)
                    self.lip_feature_encoder = torch.nn.DataParallel(self.lip_feature_encoder)
                self.ID_encoder.cuda()
                self.Decoder.cuda()
                self.mfcc_encoder.cuda()
                self.lip_feature_encoder.cuda()
                self.criterionL1.cuda()

        print('---------- Networks initialized -------------')

    def name(self):
        return 'GenModel'

    def set_test_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']

        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_path' if AtoB else 'B_path']

        self.optimizer_G_test = torch.optim.Adam(
            list(self.ID_encoder.parameters()) +
            list(self.Decoder.parameters()),
            lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def test_train(self):

        self.test_mask = Variable(self.Tensor(self.opt.batchSize, self.opt.image_channel_size, self.opt.image_size, self.opt.image_size).fill_(1))
        self.test_mask[:, :, 170:234, 64:192] = 0.001
        self.real_A = Variable(self.input_A)
        self.mfcc_encoder.eval()

        self.image_embedding = self.ID_encoder.forward(self.real_A)
        self.lip_embeddings = self.lip_feature_encoder.forward(self.real_A)
        self.fake_B = self.Decoder.forward(self.image_embedding, self.lip_embeddings)

        self.optimizer_G_test.zero_grad()

        self.loss_G_L1 = self.criterionL1(self.fake_B * self.test_mask, self.real_A * self.test_mask) * self.opt.lambda_A

        self.loss_G_L1.backward()

        self.optimizer_G_test.step()

    def test(self):
        self.ID_encoder.eval()
        self.lip_feature_encoder.eval()
        self.Decoder.eval()
        self.mfcc_encoder.eval()
        self.real_A = Variable(self.input_A)
        self.real_videos = Variable(self.input_B)

        # compute the ID embeddings

        self.real_A_id_embedding = self.ID_encoder.forward(self.real_A)

        # extract the lip feature

        self.lip_embeddings = self.lip_feature_encoder.forward(self.real_videos)

        # generate fake images

        self.video_sequence_generation()

    def video_sequence_generation(self):
        self.lip_embeddings = self.lip_embeddings.view(-1, self.opt.sequence_length, self.opt.feature_length)
        image_gen_fakes = []
        for i in range(self.opt.sequence_length):
            image_gen_fakes_buffer = self.Decoder(self.real_A_id_embedding, self.lip_embeddings[:, i, :] * 0.8)
            image_gen_fakes.append(image_gen_fakes_buffer.view(-1, 1, self.opt.image_channel_size, self.opt.image_size, self.opt.image_size))

        self.image_gen_fakes = torch.cat(image_gen_fakes, 1)

        self.image_gen_fakes_batch = self.image_gen_fakes.view(-1, self.opt.image_channel_size, self.opt.image_size, self.opt.image_size)
        self.image_gen_fakes = self.image_gen_fakes.view(-1, self.opt.image_channel_size * (self.opt.sequence_length), self.opt.image_size, self.opt.image_size)


    def get_current_visuals(self):
        fake_B_image = self.image_gen_fakes.view(-1, self.opt.sequence_length, self.opt.image_channel_size, self.opt.image_size, self.opt.image_size)
        real_A = util.tensor2im(self.real_A.data)
        oderdict = OrderedDict([('real_A', real_A)])
        fake_image_B = {}
        real_B = {}
        for i in range(self.opt.sequence_length):
            fake_image_B[i] = util.tensor2im(fake_B_image[:, i, :, :, :].data)
            real_B[i] = util.tensor2im(self.real_videos[:, i, :, :, :].data)
            oderdict['real_B_' + str(i)] = real_B[i]
            oderdict['fake_image_B_' + str(i)] = fake_image_B[i]

        return oderdict

    def get_image_paths(self):
        return self.image_paths
