import time
from Options_all import BaseOptions
from util import util
from util.visualizer import Visualizer
from torch.utils.data import DataLoader
import os
import ntpath

opt = BaseOptions().parse()

if opt.test_type == 'video' or opt.test_type == 'image':
    import Test_Gen_Models.Test_Video_Model as Gen_Model
    from Dataloader.Test_load_video import Test_VideoFolder
elif opt.test_type == 'audio':
    import Test_Gen_Models.Test_Audio_Model as Gen_Model
    from Dataloader.Test_load_audio import Test_VideoFolder
else:
    raise('test type select error')

opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.sequence_length = 1
test_nums = [1, 2, 3, 4]  # choose input identity images

model = Gen_Model.GenModel(opt)
# _, _, start_epoch = util.load_test_checkpoint(opt.test_resume_path, model)
start_epoch = opt.start_epoch
visualizer = Visualizer(opt)
# find the checkpoint's path name without the 'checkpoint.pth.tar'
path_name = ntpath.basename(opt.test_resume_path)[:-19]
web_dir = os.path.join(opt.results_dir, path_name, '%s_%s' % ('test', start_epoch))
for i in test_nums:
    A_path = os.path.join(opt.test_A_path, 'test_sample' + str(i) + '.jpg')
    test_folder = Test_VideoFolder(root=opt.test_root, A_path=A_path, config=opt)
    test_dataloader = DataLoader(test_folder, batch_size=1,
                                shuffle=False, num_workers=1)
    model, _, start_epoch = util.load_test_checkpoint(opt.test_resume_path, model)

    # inference during test

    for i2, data in enumerate(test_dataloader):
        if i2 < 5:
            model.set_test_input(data)
            model.test_train()

    # test
    start = time.time()
    for i3, data in enumerate(test_dataloader):
        model.set_test_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        visualizer.save_images_test(web_dir, visuals, img_path, i3, opt.test_num)
    end = time.time()
    print('finish processing in %03f seconds' % (end - start))

