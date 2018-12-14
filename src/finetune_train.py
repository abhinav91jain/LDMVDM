import Torch
from torch import optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.io as sio
import os
from collections import OrderedDict
from timeit import default_timer as timer
from KittiDataset import KittiDataset
from options.train_options import TrainOptions
from FinetuneLearner import finetuneLearner

opt = TrainOptions().parse()
imgSize = [opt.imH, opt.imW]

dataset = KittiDataset("train.txt", opt.dataroot, imgSize, 3)
dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.nThreads, pin_memory=True)

gpu_ids = list(range(opt.batchSize))

finelearner = finetuneLearner(imgSize = imgSize, refIdx=1, S=opt.lambda_S, gpuIds=gpu_ids, smoothTerm=opt.smooth_term, use_ssim=opt.use_ssim)
finelearner.init_weights()

if opt.which_epoch >= 0:
	print("last checkpoint")
	finelearner.load_model(os.path.join(opt.checkpoints_dir, str(opt.which_epoch)+"_model.pth"), os.path.join(opt.checkpoints_dir, 'pose_net.pth'))
else:
	print("pretrained model")
	finelearner.load_model(os.path.join(opt.checkpoints_dir, 'depth_net.pth'), os.path.join(opt.checkpoints_dir, 'pose_net.pth'))

finelearner.cuda()

def visDepthMap(input):
	x = (input-input.min()) * (255/(input.max()-input.min()+.00001))
    return x.unsqueeze(2).repeat(1, 1, 3)

optimizer = optim.Adam(finelearner.get_parameters(), lr=.0001)

step_num = 0

for epoch in range(max(0, opt.which_epoch), opt.epoch_num+1):
	t = timer()
	for ii, data in enumerate(dataloader):
        optimizer.zero_grad()
        frames = Variable(data[0].float().cuda())
        camparams = Variable(data[1])
        cost, photometric_cost, smoothness_cost, frames, inv_depths = finelearner.forward(frames, camparams, iterNum=opt.max_lk_iter_num, level=opt.lk_level)
        cost_ = cost.data.cpu()
        inv_depths_mean = inv_depths.mean().data.cpu().numpy()
        cost.backward()
        optimizer.step()

        step_num+=1

        if np.mod(step_num, opt.print_freq)==0:
            elapsed_time = timer()-t
            print('%s: %s / %s, ... elapsed time: %f (s)' % (epoch, step_num, int(len(dataset)/opt.batchSize), elapsed_time))
            print(inv_depths_mean)
            t = timer()

        if np.mod(step_num, opt.display_freq)==0:
            frame_vis = frames.data.permute(1,2,0).contiguous().cpu().numpy().astype(np.uint8)
            depth_vis = vis_depthmap(inv_depths.data.cpu()).numpy().astype(np.uint8)
            sio.savemat(os.path.join(opt.checkpoints_dir, 'depth_' + str(step_num) + '.mat',
                {'D': inv_depths.data.cpu().numpy(),
                 'I': frame_vis})

        if np.mod(step_num, opt.save_latest_freq)==0:
            print("cache model....")
            finelearner.save_model(os.path.join(opt.checkpoints_dir, str(epoch) + '_model.pth'))
            finelearner.cuda()
            print('..... saved')