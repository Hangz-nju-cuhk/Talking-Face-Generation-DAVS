"""
This model only disentangles pid from wid inside the IdentityEncoder, which is the most crucial part
"""
from __future__ import print_function, division

from collections import OrderedDict

import torch
from torch.autograd import Variable
import numpy as np
import Options
import random
import embedding_utils
import loss_functions
import network.FAN_feature_extractor as FAN_feature_extractor
import network.IdentityEncoder as IdentityEncoder
# import network.VGGM as VGGM
import network.Decoder_networks as Decoder_network
import network.mfcc_networks as mfcc_networks
import network.networks as networks
import util.util as util
from network import Discriminator_networks as Discriminator_networks

# opt = Options.Config()


class GenModel():

    def __init__(self, opt):
        self.opt = opt
        self.Tensor = torch.cuda.FloatTensor if opt.cuda_on else torch.Tensor
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.image_channel_size,
                                   opt.image_size, opt.image_size)
        self.input_B = self.Tensor(opt.batchSize, opt.pred_length, opt.image_channel_size,
                                   opt.image_size, opt.image_size)
        self.input_video = self.Tensor(opt.batchSize, opt.sequence_length + 1, opt.image_channel_size,
                                   opt.image_size, opt.image_size)
        self.input_audio = self.Tensor(opt.batchSize, opt.sequence_length + 1, 1,
                                       opt.mfcc_length, opt.mfcc_width)
        self.B_audio = self.Tensor(opt.batchSize, opt.pred_length, 1,
                                       opt.mfcc_length, opt.mfcc_width)
        self.input_video_dis = self.Tensor(opt.batchSize, opt.disfc_length , opt.image_channel_size,
                                   opt.image_size, opt.image_size)
        self.video_pred_data = self.Tensor(opt.batchSize, opt.pred_length, opt.image_channel_size,
                                   opt.image_size, opt.image_size)
        self.audio_pred_data = self.Tensor(opt.batchSize, opt.pred_length, 1,
                                   opt.image_size, opt.image_size)

        self.ID_encoder = IdentityEncoder.IdentityEncoder()

        self.Decoder = Decoder_network.Decoder(opt)

        # audio wid feature encoder
        self.mfcc_encoder = mfcc_networks.mfcc_encoder_two(opt)

        # visual wid feature encoder
        self.lip_feature_encoder = FAN_feature_extractor.FanFusion(opt)

        # discriminator to disentangle wid from pid
        self.ID_lip_discriminator = Discriminator_networks.ID_dis32(feature_length=64, config=opt)

        # Classifier from wid to class label
        self.model_fusion = networks.ModelFusion(opt)

        # discriminator for adv in embedding wid
        self.discriminator_audio = networks.discriminator_audio()

        use_sigmoid = opt.no_lsgan
        self.netD = Discriminator_networks.Discriminator(input_nc=3, use_sigmoid=use_sigmoid)
        self.netD_mul = Discriminator_networks.Discriminator(input_nc=3 * opt.sequence_length, use_sigmoid=use_sigmoid)
        self.netD_mul.apply(networks.weights_init)
        self.netD.apply(networks.weights_init)
        # self.Decoder.apply(networks.weights_init)
        self.ID_lip_discriminator.apply(networks.weights_init)

        self.old_lr = opt.lr
        # define loss functions
        self.criterionGAN = loss_functions.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor, softlabel=False)
        self.criterionGAN_soft = loss_functions.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor, softlabel=True)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionSmoothL1 = torch.nn.SmoothL1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.L2Contrastive = loss_functions.L2ContrastiveLoss(margin=opt.L2margin)
        self.criterionCE = torch.nn.CrossEntropyLoss()
        self.inv_dis_loss = loss_functions.L2SoftmaxLoss()


        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(list(self.Decoder.parameters()) +
                                            list(self.ID_encoder.parameters()) +
                                            list(self.model_fusion.parameters()) +
                                            list(self.mfcc_encoder.parameters()) +
                                            list(self.lip_feature_encoder.parameters()),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(list(self.netD.parameters()) + list(self.netD_mul.parameters()) +
                                            list(self.discriminator_audio.parameters()) +
                                            list(self.ID_lip_discriminator.parameters()) ,
                                            lr=opt.lr, betas=(opt.beta1, 0.999))

        if torch.cuda.is_available():
            if opt.cuda_on:
                if opt.mul_gpu:
                    self.ID_encoder = torch.nn.DataParallel(self.ID_encoder)
                    self.Decoder = torch.nn.DataParallel(self.Decoder)
                    self.mfcc_encoder = torch.nn.DataParallel(self.mfcc_encoder)
                    self.netD_mul = torch.nn.DataParallel(self.netD_mul)
                    self.netD = torch.nn.DataParallel(self.netD)
                    self.lip_feature_encoder = torch.nn.DataParallel(self.lip_feature_encoder)
                    self.ID_lip_discriminator = torch.nn.DataParallel(self.ID_lip_discriminator)
                    self.model_fusion = torch.nn.DataParallel(self.model_fusion)
                    self.discriminator_audio = torch.nn.DataParallel(self.discriminator_audio)
                self.ID_encoder.cuda()
                self.Decoder.cuda()
                self.mfcc_encoder.cuda()
                self.lip_feature_encoder.cuda()
                self.ID_lip_discriminator.cuda()
                self.netD_mul.cuda()
                self.netD.cuda()
                self.criterionL1.cuda()
                self.criterionGAN.cuda()
                self.criterionGAN_soft.cuda()
                self.criterionL2.cuda()
                self.criterionCE.cuda()
                self.inv_dis_loss.cuda()
                self.model_fusion.cuda()
                self.discriminator_audio.cuda()
                self.L2Contrastive.cuda()

        print('---------- Networks initialized -------------')

    def name(self):
        return 'GenModel'

    def set_input(self, input, input_label):
        input_video = input['video']
        input_audio = input['mfcc20']
        self.input_label = input_label.cuda()
        dis_select_start = random.randint(0, 25 - self.opt.disfc_length - 1)
        A_select = random.randint(0, 28)
        pred_start = random.randint(0, 1)
        input_A = input_video[:, A_select, :, :, :].contiguous()
        input_video_dis = input_video[:, dis_select_start:dis_select_start + self.opt.disfc_length, :, :, :]
        video_pred_data = input_video[:, pred_start:pred_start + self.opt.pred_length * 2:2, :, :, :]
        audio_pred_data = input_audio[:, pred_start:pred_start + self.opt.pred_length * 2:2, :, :, :]
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_video_dis.resize_(input_video_dis.size()).copy_(input_video_dis)
        self.video_pred_data.resize_(video_pred_data.size()).copy_(video_pred_data)
        self.audio_pred_data.resize_(audio_pred_data.size()).copy_(audio_pred_data)
        self.image_paths = input['A_path']

    def forward(self):

        self.input_label = Variable(self.input_label)
        self.real_A = Variable(self.input_A)
        B_start = random.randint(0, self.opt.pred_length - self.opt.sequence_length)
        self.audios_dis = Variable(self.audio_pred_data)
        self.video_dis = Variable(self.video_pred_data)
        # real_videos are the frames used for training generation,
        self.real_videos = Variable(self.video_pred_data[:, B_start:B_start + self.opt.sequence_length, :, :, :].contiguous())
        self.audios = Variable(self.audio_pred_data[:, B_start:B_start + self.opt.sequence_length, :, :, :].contiguous())
        self.video_send_to_disfc = Variable(self.input_video_dis)
        self.mask = Variable(self.Tensor(self.opt.batchSize, (self.opt.sequence_length) * self.opt.image_channel_size, self.opt.image_size, self.opt.image_size).fill_(0))
        self.mask[:, :, 170:234, 64:192] = 1
        self.mask_ones = Variable(self.Tensor(self.opt.batchSize, self.opt.image_channel_size, self.opt.image_size,
                                              self.opt.image_size).fill_(1))
        self.mask_ones[:, :, 170:234, 64:192] = 0
        self.mfcc_encoder.train()
        self.lip_feature_encoder.train()

        # compute the ID embeddings
        self.real_A_id_embedding = self.ID_encoder.forward(self.real_A)

        # compute the sequence ID imbeddings
        if self.opt.disfc_length == 12:
            self.sequence_id_embedding = self.ID_encoder.forward(self.video_dis)
        else:
            self.sequence_id_embedding = self.ID_encoder.forward(self.video_send_to_disfc)
        self.sequence_id_embedding = self.sequence_id_embedding[4].view(-1, self.opt.disfc_length * 64, 64, 64)

        # extract the lip feature

        # self.audio_embedding = self.mfcc_encoder.forward(self.audio_A)
        self.audio_embeddings_dis = self.mfcc_encoder.forward(self.audios_dis)
        self.lip_embeddings_dis = self.lip_feature_encoder.forward(self.video_dis)
        self.audio_embeddings = self.audio_embeddings_dis[:, B_start:B_start + self.opt.sequence_length].contiguous()
        self.lip_embeddings = self.lip_embeddings_dis[:, B_start:B_start + self.opt.sequence_length].contiguous()

        # loss between audio and lip embedding
        self.lip_embedding_norm = embedding_utils.l2_norm(self.lip_embeddings_dis.view(-1, 256 * self.opt.pred_length))
        self.audio_embedding_norm = embedding_utils.l2_norm(self.audio_embeddings_dis.view(-1, 256 * self.opt.pred_length))
        self.lip_embeddings_buffer = Variable(self.lip_embedding_norm.data)
        self.EmbeddingL2 = self.L2Contrastive.forward(self.lip_embeddings_buffer, self.audio_embedding_norm)
        # generate fake images

        self.sequence_generation()

        # single
        self.fakes = torch.cat((self.audio_gen_fakes_batch, self.image_gen_fakes_batch), 0)
        self.real_one = self.real_videos.view(-1, self.opt.image_channel_size, self.opt.image_size, self.opt.image_size)
        self.reals = torch.cat((self.real_one, self.real_one), 0)
        self.audio_reals = torch.cat((self.audios.view(-1, 1, self.opt.mfcc_length, self.opt.mfcc_width),
                                      self.audios.view(-1, 1, self.opt.mfcc_length, self.opt.mfcc_width)), 0)

        # sequence
        self.fakes_sequence = self.fakes.view(-1, self.opt.image_channel_size * (self.opt.sequence_length), self.opt.image_size, self.opt.image_size)
        self.real_one_sequence = self.real_videos.view(-1, self.opt.image_channel_size * (self.opt.sequence_length), self.opt.image_size, self.opt.image_size)
        self.reals_sequence = self.reals.view(-1, self.opt.image_channel_size * self.opt.sequence_length, self.opt.image_size, self.opt.image_size)
        self.audio_reals_sequence = self.audio_reals.view(-1, self.opt.sequence_length, self.opt.mfcc_length, self.opt.mfcc_width)


    def sequence_generation(self):
        self.lip_embeddings = self.lip_embeddings.view(-1, self.opt.sequence_length, self.opt.feature_length)
        image_gen_fakes = []
        self.audio_embeddings = self.audio_embeddings.view(-1, self.opt.sequence_length, self.opt.feature_length)
        audio_gen_fakes = []
        self.last_frame = Variable(self.real_A.data)
        self.G_x_loss = 0
        for i in range(self.opt.sequence_length):
            image_gen_fakes_buffer = self.Decoder(self.real_A_id_embedding, self.lip_embeddings[:, i, :])
            image_gen_fakes.append(image_gen_fakes_buffer.view(-1, 1, self.opt.image_channel_size, self.opt.image_size, self.opt.image_size))
            audio_gen_fakes_buffer = self.Decoder(self.real_A_id_embedding, self.audio_embeddings[:, i, :])
            audio_gen_fakes.append(audio_gen_fakes_buffer.view(-1, 1, self.opt.image_channel_size, self.opt.image_size, self.opt.image_size))
            self.G_x_loss = self.G_x_loss + self.criterionL1(audio_gen_fakes_buffer* self.mask_ones, self.last_frame * self.mask_ones)
            last_frame = audio_gen_fakes_buffer.data
            self.last_frame = Variable(last_frame)
            if i > 0:
                last_frame = audio_gen_fakes_buffer.data
                self.last_frame = Variable(last_frame)

        self.image_gen_fakes = torch.cat(image_gen_fakes, 1)

        self.image_gen_fakes_batch = self.image_gen_fakes.view(-1, self.opt.image_channel_size, self.opt.image_size, self.opt.image_size)
        self.image_gen_fakes = self.image_gen_fakes.view(-1, self.opt.image_channel_size * (self.opt.sequence_length), self.opt.image_size, self.opt.image_size)
        self.audio_gen_fakes = torch.cat(audio_gen_fakes, 1)

        self.audio_gen_fakes_batch = self.audio_gen_fakes.view(-1, self.opt.image_channel_size, self.opt.image_size, self.opt.image_size)
        self.audio_gen_fakes = self.audio_gen_fakes.view(-1, self.opt.image_channel_size * (self.opt.sequence_length), self.opt.image_size, self.opt.image_size)

    def backward_dis(self):
        self.audio_D_real = self.discriminator_audio(self.audio_embeddings_dis.detach())
        self.audio_D_fake = self.discriminator_audio(self.lip_embeddings_dis.detach())
        self.image_loss_D_real = self.criterionGAN(self.audio_D_fake, False)
        self.audio_loss_D_real = self.criterionGAN(self.audio_D_real, True)
        self.dis_R_loss = (self.image_loss_D_real + self.audio_loss_D_real) * 0.5
        self.dis_R_loss.backward()

    def backward_D(self):

        # train ID_disciminate_fc

        self.lip_pred = self.ID_lip_discriminator(self.sequence_id_embedding.detach())
        self.CE_loss = self.criterionCE(self.lip_pred, self.input_label) * self.opt.lambda_CE

        # GAN single fake
        if self.opt.require_single_GAN:
            self.pred_fake_single, self.pred_fake_single_combine = self.netD.forward(self.fakes.detach(), self.audio_reals)
            self.loss_D_single_fake = self.criterionGAN_soft(self.pred_fake_single, False)
            self.loss_D_single_combine_fake = self.criterionGAN_soft(self.pred_fake_single_combine, False)

        # GAN single real

            self.pred_real, self.pred_real_combine = self.netD.forward(self.reals, self.audio_reals)
            self.loss_D_single_real = self.criterionGAN_soft(self.pred_real, True)
            self.loss_D_single_combine_real = self.criterionGAN_soft(self.pred_real_combine, True)

            self.loss_D_single = (self.loss_D_single_fake + self.loss_D_single_real) * 0.5
            self.loss_D_single_combine = (self.loss_D_single_combine_fake + self.loss_D_single_combine_real) * 0.5
        else:
            self.loss_D_single_combine = 0
            self.loss_D_single = 0


        if self.opt.require_sequence_GAN:
            # GAN sequence fake
            self.pred_fake_sequence, self.pred_fake_sequence_combine = self.netD_mul.forward(self.fakes_sequence.detach(), self.audio_reals_sequence)
            self.loss_D_sequence_fake = self.criterionGAN_soft(self.pred_fake_sequence, False)
            self.loss_D_sequence_combine_fake = self.criterionGAN_soft(self.pred_fake_sequence_combine, False)

            # GAN sequence real

            self.pred_real_sequence, self.pred_real_sequence_combine = self.netD_mul.forward(self.reals_sequence, self.audio_reals_sequence)
            self.loss_D_sequence_real = self.criterionGAN_soft(self.pred_real_sequence, True)
            self.loss_D_sequence_combine_real = self.criterionGAN_soft(self.pred_real_sequence_combine, True)

            self.loss_D_sequence = (self.loss_D_sequence_fake + self.loss_D_sequence_real) * 0.5
            self.loss_D_sequence_combine = (self.loss_D_sequence_combine_fake + self.loss_D_sequence_combine_real) * 0.5

        else:
            self.loss_D_sequence_combine = 0
            self.loss_D_sequence = 0


        # Combined loss

        self.loss_D = (self.loss_D_sequence_combine + self.loss_D_sequence) + \
                      (self.loss_D_single_combine + self.loss_D_single) + \
                      self.CE_loss

        self.loss_D.backward()

    def backward_G(self):

        self.audio_D_real = self.discriminator_audio(self.audio_embeddings_dis)
        self.audio_loss_D_inv = self.criterionGAN(self.audio_D_real, False)
        self.audio_D_fake = self.discriminator_audio(self.lip_embeddings_dis)
        self.image_loss_D_inv = self.criterionGAN(self.audio_D_fake, True)

        # classification
        self.audio_pred = self.model_fusion.forward(self.audio_embeddings_dis)
        self.audio_CE_loss = self.criterionCE(self.audio_pred, self.input_label)
        self.audio_acc = self.compute_acc(self.audio_pred)
        self.image_pred = self.model_fusion.forward(self.lip_embeddings_dis)
        self.image_CE_loss = self.criterionCE(self.image_pred, self.input_label)
        self.image_acc = self.compute_acc(self.image_pred)


        # id_discriminator
        self.lip_pred = self.ID_lip_discriminator(self.sequence_id_embedding)
        self.softmax_loss = self.inv_dis_loss.forward(self.lip_pred) * self.opt.lambda_CE_inv

        self.lip_acc = self.compute_acc(self.lip_pred)
        # single
        if self.opt.require_single_GAN:
            pred_fake, pred_combine_fake = self.netD.forward(self.fakes, self.audio_reals)
            self.loss_G_GAN_single = self.criterionGAN(pred_fake, True)
            self.loss_G_GAN_single_combine = self.criterionGAN(pred_combine_fake, True)

        else:
            self.loss_G_GAN_single = 0
            self.loss_G_GAN_single_combine = 0

        #sequence
        if self.opt.require_sequence_GAN:
            pred_fake, pred_combine_fake = self.netD_mul.forward(self.fakes_sequence, self.audio_reals_sequence)
            self.loss_G_GAN_sequence = self.criterionGAN(pred_fake, True)
            self.loss_G_GAN_sequence_combine = self.criterionGAN(pred_combine_fake, True)

        else:
            self.loss_G_GAN_sequence = 0
            self.loss_G_GAN_sequence_combine = 0

        self.loss_G_L1_audio = self.criterionL1(self.audio_gen_fakes * 255, self.real_one_sequence * 255) * self.opt.lambda_A + \
                         self.criterionL1(self.audio_gen_fakes * self.mask * 255, self.real_one_sequence * self.mask * 255) * self.opt.lambda_B
        self.loss_G_L1_image = self.criterionL1(self.image_gen_fakes * 255, self.real_one_sequence * 255) * self.opt.lambda_A + \
                         self.criterionL1(self.image_gen_fakes * self.mask * 255, self.real_one_sequence * self.mask * 255) * self.opt.lambda_B

        self.loss_G = (self.loss_G_GAN_single + self.loss_G_GAN_single_combine) + \
                      (self.loss_G_GAN_sequence + self.loss_G_GAN_sequence_combine) + \
                      self.loss_G_L1_audio + self.loss_G_L1_image + self.G_x_loss * 5\
                      + self.EmbeddingL2 +\
                      self.softmax_loss + self.audio_CE_loss + self.image_CE_loss + \
                      (self.audio_loss_D_inv + self.image_loss_D_inv)*5

        self.loss_G.backward()

    def set_test_input(self, input, input_label):
        input_video = input['video']
        input_audio = input['mfcc20']
        self.input_label = input_label.cuda()
        dis_select_start = random.randint(0, 25 - self.opt.disfc_length - 1)
        pred_start = random.randint(0, 1)
        input_video_dis = input_video[:, dis_select_start:dis_select_start + self.opt.disfc_length, :, :, :]
        video_pred_data = input_video[:, pred_start:pred_start + self.opt.pred_length * 2:2, :, :, :]
        audio_pred_data = input_audio[:, pred_start:pred_start + self.opt.pred_length * 2:2, :, :, :]
        self.input_video_dis.resize_(input_video_dis.size()).copy_(input_video_dis)
        self.video_pred_data.resize_(video_pred_data.size()).copy_(video_pred_data)
        self.audio_pred_data.resize_(audio_pred_data.size()).copy_(audio_pred_data)
        self.image_paths = input['A_path']

    def test(self):
        self.mfcc_encoder.eval()
        self.lip_feature_encoder.eval()
        self.input_label = Variable(self.input_label, volatile=True)
        self.audios_dis = Variable(self.audio_pred_data, volatile=True)
        self.video_dis = Variable(self.video_pred_data, volatile=True)

        # compute the sequence ID imbeddings

        self.audio_embeddings_dis = self.mfcc_encoder.forward(self.audios_dis).view(-1, 256 * self.opt.pred_length)
        self.lip_embeddings_dis = self.lip_feature_encoder.forward(self.video_dis).view(-1, 256 * self.opt.pred_length)

        # loss between audio and lip embedding
        self.lip_embedding_norm = embedding_utils.l2_norm(self.lip_embeddings_dis)
        self.audio_embedding_norm = embedding_utils.l2_norm(self.audio_embeddings_dis)
        self.lip_embeddings_buffer = Variable(self.lip_embedding_norm.data)
        self.EmbeddingL2 = self.L2Contrastive.forward(self.audio_embedding_norm, self.lip_embeddings_buffer)
        # generate fake images
        # classification
        self.audio_pred = self.model_fusion.forward(self.audio_embeddings_dis)
        self.audio_acc = self.compute_acc(self.audio_pred)
        self.image_pred = self.model_fusion.forward(self.lip_embeddings_dis)
        self.image_acc = self.compute_acc(self.image_pred)
        self.output = (self.audio_pred + self.image_pred)
        self.final_acc = self.compute_acc(self.output)

    def forward_no_generation(self):
    # Used when training without generation
        self.mfcc_encoder.train()
        self.lip_feature_encoder.train()
        self.model_fusion.train()
        self.input_audio_data = Variable(self.audio_pred_data)
        self.input_image_data = Variable(self.video_pred_data)
        self.input_label = Variable(self.input_label)

        self.audio_embeddings_dis = self.mfcc_encoder.forward(self.input_audio_data)
        self.lip_embeddings_dis = self.lip_feature_encoder.forward(self.input_image_data)
        self.audio_fusion = self.audio_embeddings_dis.view(-1, int(256 * self.opt.pred_length / 3))
        self.image_fusion = self.lip_embeddings_dis.view(-1, int(256 * self.opt.pred_length / 3))
        self.audio_fusion_buffer = Variable(self.audio_fusion.data)
        self.image_fusion_buffer = Variable(self.image_fusion.data)

    def backward_no_generation(self):

    # Used when training without generation

        self.audio_D_real = self.discriminator_audio(self.audio_embeddings_dis)

        self.audio_loss_D_inv = self.criterionGAN(self.audio_D_real, False)

        self.audio_output = self.model_fusion.forward(self.audio_embeddings_dis)
        self.audio_CE_loss = self.criterionCE(self.audio_output, self.input_label)
        self.audio_L2_loss = self.L2Contrastive.forward(self.image_fusion.detach(), self.audio_fusion)
        self.audio_ranking_loss = self.Contrastive.forward(self.image_fusion.detach(), self.audio_fusion)
        self.loss_audio = self.audio_loss_D_inv + self.audio_CE_loss + \
                          self.audio_L2_loss
        self.loss_audio.backward()

        audio_D_fake = self.discriminator_audio(self.lip_embeddings_dis)
        self.image_loss_D_inv = self.criterionGAN(audio_D_fake, True)

        self.image_output = self.model_fusion.forward(self.lip_embeddings_dis)
        self.image_acc = self.compute_acc(self.image_output)
        self.output = (self.audio_pred + self.image_output)
        self.acc = self.compute_acc(self.output)
        self.image_CE_loss = self.criterionCE(self.image_output, self.input_label)
        self.image_L2_loss = self.L2Contrastive.forward(self.audio_fusion.detach(), self.image_fusion)
        self.image_ranking_loss = self.Contrastive.forward(self.audio_fusion.detach(), self.image_fusion)
        self.loss_image = self.image_L2_loss + self.image_CE_loss\
                          + self.image_loss_D_inv
        self.loss_image.backward()

    def save_feature(self):
        self.ID_encoder.eval()
        self.ID_lip_discriminator.eval()
        self.video_send_to_disfc = Variable(self.input_video_dis, volatile=True)
        # compute the sequence ID imbeddings

        # self.audio_embedding = self.mfcc_encoder.forward(self.audio_A)
        self.sequence_id_embedding = self.ID_encoder.forward(self.video_send_to_disfc)
        # self.sequence_id_embedding = self.sequence_id_embedding[0].view(-1, opt.disfc_length, opt.feature_length)
        self.lip_pred_feature = self.sequence_id_embedding[0].view(-1, self.opt.disfc_length * 256)

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_dis()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def optimize_parameters_no_generation(self):
        self.forward_no_generation()
        self.optimizer_D.zero_grad()
        self.backward_dis()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_no_generation()
        self.optimizer_G.step()


    def get_current_errors(self):
        if self.opt.require_single_GAN:
            return OrderedDict([('G_GAN_single', self.loss_G_GAN_single.data[0]),
                                ('G_GAN_single_combine', self.loss_G_GAN_single_combine.data[0]),
                                ('G_L1_audio', self.loss_G_L1_audio.data[0]),
                                ('G_L1_image', self.loss_G_L1_image.data[0]),
                                ('D_real_single', self.loss_D_single_real.data[0]),
                                ('D_fake_single', self.loss_D_single_fake.data[0]),
                                ('D_combine_real_single', self.loss_D_single_combine_real.data[0]),
                                ('D_combine_fake_single', self.loss_D_single_combine_fake.data[0]),
                                ('CE_loss', self.CE_loss.data[0]),
                                ('lossoftmax', self.softmax_loss.data[0]),
                                ('audio_acc', self.audio_acc),
                                ('image_acc', self.image_acc),
                                ('EmbeddingL2', self.EmbeddingL2.data[0]),
                                ('dis_R_loss', self.dis_R_loss.data[0])
                                ])
        else:
            return OrderedDict([('G_GAN_sequence', self.loss_G_GAN_sequence.data[0]),
                                ('G_GAN_sequence_combine', self.loss_G_GAN_sequence_combine.data[0]),
                                ('G_L1_audio', self.loss_G_L1_audio.data[0]),
                                ('G_L1_image', self.loss_G_L1_image.data[0]),
                                ('D_real_sequence', self.loss_D_sequence_real.data[0]),
                                ('D_fake_sequence', self.loss_D_sequence_combine_real.data[0]),
                                ('D_combine_real_sequence', self.loss_D_sequence_combine_real.data[0]),
                                ('D_combine_fake_sequence', self.loss_D_sequence_combine_fake.data[0]),
                                ('CE_loss', self.CE_loss.data[0]),
                                ('lossoftmax', self.softmax_loss.data[0]),
                                ('audio_acc', self.audio_acc),
                                ('image_acc', self.image_acc),
                                ('EmbeddingL2', self.EmbeddingL2.data[0]),
                                ('dis_R_loss', self.dis_R_loss.data[0])
                                ])

    def get_current_visuals(self):
        fake_B_audio = self.audio_gen_fakes.view(-1, self.opt.sequence_length, self.opt.image_channel_size, self.opt.image_size, self.opt.image_size)
        fake_B_image = self.image_gen_fakes.view(-1, self.opt.sequence_length, self.opt.image_channel_size, self.opt.image_size, self.opt.image_size)
        real_A = util.tensor2im(self.real_A.data)
        oderdict = OrderedDict([('real_A', real_A)])
        fake_audio_B = {}
        fake_image_B = {}
        real_B = {}
        for i in range(self.opt.sequence_length):
            fake_audio_B[i] = util.tensor2im(fake_B_audio[:, i, :, :, :].data)
            fake_image_B[i] = util.tensor2im(fake_B_image[:, i, :, :, :].data)
            real_B[i] = util.tensor2im(self.real_videos[:, i, :, :, :].data)
            oderdict['real_B_' + str(i)] = real_B[i]
            oderdict['fake_audio_B_' + str(i)] = fake_audio_B[i]
            oderdict['fake_image_B_' + str(i)] = fake_image_B[i]

        return oderdict

    def get_visual_path(self):
        print(self.image_paths[0])

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def compute_acc(self, out):
        _, pred = out.topk(1, 1)
        pred0 = pred.squeeze().data
        acc = 100 * torch.sum(pred0 == self.input_label.data) / self.input_label.size(0)
        return acc

    def TfWriter(self, writer, total_steps):
        # write loss to tensorboard
        writer.add_scalar('train_image_L2_loss', embedding_utils.to_np(self.EmbeddingL2), total_steps)
        writer.add_scalar('image_loss_D_inv', embedding_utils.to_np(self.image_loss_D_inv), total_steps)
        writer.add_scalar('train_audio_acc', self.audio_acc, total_steps)
        writer.add_scalar('train_image_acc', self.image_acc, total_steps)






