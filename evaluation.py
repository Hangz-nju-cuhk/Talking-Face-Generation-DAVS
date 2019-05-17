from __future__ import print_function, division
import numpy as np
import Options
import embedding_utils as util

config = Options.Config()


def evaluation(eval_dataloader, audio_model, total_steps, writer):

    eval_steps = len(eval_dataloader)
    print(eval_steps)
    # set model to train model
    ACC = 0
    audio_ACC = 0
    image_ACC = 0

    avg_eval_loss = 0
    avg_L2_eval_loss = 0
    avg_ranking_loss = 0
    audio_ebds = []
    image_ebds = []
    for eval_step, (eval_data, eval_labels) in enumerate(eval_dataloader):
        # load training data

        # forward the data into model
        audio_model.set_test_input(eval_data, eval_labels)
        audio_model.test()
        audio_ebds.append(util.to_np(audio_model.audio_embedding_norm))
        image_ebds.append(util.to_np(audio_model.lip_embedding_norm))
        avg_L2_eval_loss += util.to_np(audio_model.EmbeddingL2)
        audio_ACC = audio_model.audio_acc + audio_ACC
        image_ACC = audio_model.image_acc + image_ACC
        ACC = audio_model.final_acc + ACC

    audio_ebds = np.concatenate(audio_ebds, axis=0)
    image_ebds = np.concatenate(image_ebds, axis=0)
    metrics = util.L2retrieval(audio_ebds, image_ebds)
    metrics_inv = util.L2retrieval(image_ebds, audio_ebds)
    # -- print log
    info = 'Video Retrieval ({} samples): R@1: {:.2f}, R@5: {:.2f}, R@10: {:.2f}, R@50: {:.2f}, MedR: {:.1f}, MeanR: {:.1f}'
    info_inv = 'Audio Retrieval ({} samples): R@1: {:.2f}, R@5: {:.2f}, R@10: {:.2f}, R@50: {:.2f}, MedR: {:.1f}, MeanR: {:.1f}'

    ACC = ACC / eval_steps
    audio_ACC /= eval_steps
    image_ACC /= eval_steps

    avg_L2_eval_loss = avg_L2_eval_loss / eval_steps
    avg_eval_loss = avg_eval_loss / eval_steps
    print('Val L2 loss is %f' % avg_L2_eval_loss)
    print('Val loss is %f' % avg_eval_loss)
    print('Val audio accuracy is %f' % audio_ACC)
    print('Val image accuracy is %f' % image_ACC)
    print('Val accuracy is %f' % ACC)
    print(info.format(audio_ebds.shape[0], *metrics))
    print(info_inv.format(image_ebds.shape[0], *metrics_inv))

    writer.add_scalar('val_L2_loss', avg_L2_eval_loss, total_steps)
    # writer.add_scalar('val_loss', avg_eval_loss, audio_model.start_step)
    writer.add_scalar('val_acc', ACC, total_steps)
    writer.add_scalar('val_audio_acc', audio_ACC, total_steps)
    writer.add_scalar('val_image_acc', image_ACC, total_steps)
    writer.add_scalar('val_ranking_loss', avg_ranking_loss, total_steps)
    writer.add_scalar('val_retrieval top10', metrics[2], total_steps)
