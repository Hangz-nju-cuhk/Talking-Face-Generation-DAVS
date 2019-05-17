import time
from Options import Config
from Dataloader.Data_load_sequence import VideoFolder
from util import util
from util.visualizer import Visualizer
from torch.utils.data import Dataset, DataLoader
import Gen_final_v1 as Gen_Model
from evaluation import evaluation
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import os


def to_np(x):
    return x.data.cpu().numpy()


opt = Config().parse()

writer = SummaryWriter(comment=opt.name)

train_data_path = os.path.join(opt.main_PATH, 'train')
train_videoloader = VideoFolder(root=train_data_path, mode='train')
train_dataloader = DataLoader(train_videoloader, batch_size=opt.batchSize,
                            shuffle=True, num_workers=opt.num_workers, drop_last=True)

val_data_path = os.path.join(opt.main_PATH, 'val')
val_videoloader = VideoFolder(root=val_data_path, mode='val')
val_dataloader = DataLoader(val_videoloader, batch_size=opt.batchSize,
                            shuffle=True, num_workers=opt.num_workers, drop_last=True)

dataset_size = len(train_videoloader)

print('#training images = %d' % len(train_videoloader))
start_epoch = 0
total_steps = 0
model = Gen_Model.GenModel(opt)
if opt.resume:
    model, start_step, start_epoch = util.load_checkpoint(opt.resume_path, model)
    total_steps = start_step
else:
    model = util.load_separately(opt, model)
visualizer = Visualizer(opt)

cudnn.benchmark = True

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    for i, (data, label) in enumerate(train_dataloader):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data, label)
        model.optimize_parameters()
        model.TfWriter(writer, total_steps)

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)
            model.get_visual_path()

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if epoch_iter % opt.eval_freq == 0:
            evaluation(val_dataloader, model, total_steps, writer=writer)

        if total_steps % opt.save_latest_freq == 0:
            print(opt.name + 'saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            loss = to_np(model.loss_G)
            util.save_checkpoint({
                'step': total_steps,
                'epoch': epoch,
                'mfcc_encoder': model.mfcc_encoder.state_dict(),
                'lip_feature_encoder': model.lip_feature_encoder.state_dict(),
                'Decoder': model.Decoder.state_dict(),
                'ID_encoder': model.ID_encoder.state_dict(),
                'ID_lip_discriminator': model.ID_lip_discriminator.state_dict(),
                'netD': model.netD.state_dict(),
                'netD_mul': model.netD_mul.state_dict(),
                'optimizer_D': model.optimizer_D.state_dict(),
                'optimizer_G': model.optimizer_G.state_dict(),
                'model_fusion': model.model_fusion.state_dict(),
                'discriminator_audio': model.discriminator_audio.state_dict()
            }, epoch)

    test_data_path = os.path.join(opt.main_PATH, 'test')
    test_videoloader = VideoFolder(root=test_data_path, mode='test')
    test_dataloader = DataLoader(test_videoloader, batch_size=opt.batchSize,
                                 shuffle=True, num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    evaluation(test_dataloader, model, total_steps, writer=writer)

    print(opt.name + ' End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()


