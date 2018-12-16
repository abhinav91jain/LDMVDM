import torch
from torch import optim
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.io as sio
import os

from CNNLearner import CNNLearner
from KittiDataset import KittiDataset

from collections import OrderedDict
from options.train_options import TrainOptions


opt = TrainOptions().parse()
img_size = [opt.imH, opt.imW]

dataset = KittiDataset(dataPath=opt.dataroot, imgSize=img_size, bundleSize=3)
dataloader = DataLoader(dataset, batch_size=opt.batchSize,shuffle=True, num_workers=opt.nThreads, pin_memory=True)

gpu_ids = list(range(opt.batchSize))


cnnlearner = CNNLearner(img_size=img_size, ref_frame_idx=1, l1=opt.lambda_S, gpu_ids = gpu_ids, smooth_term = opt.smooth_term, use_ssim=opt.use_ssim)
cnnlearner.init_weights()


if opt.which_epoch >= 0:
    print("load pretrained model")
    cnnlearner.load_model(os.path.join(opt.checkpoints_dir, '%s' % (opt.which_epoch)))

cnnlearner.cuda()

ref_frame_idx = 1


def vis_depthmap(input):
    x = (input-input.min()) * (255/(input.max()-input.min()+.00001))
    return x.unsqueeze(2).repeat(1, 1, 3)


optimizer = optim.Adam(cnnlearner.get_parameters(), lr=.0001)

step_num = 0



for epoch in range(max(0, opt.which_epoch), opt.epoch_num+1):
    for ii, data in enumerate(dataloader):
        optimizer.zero_grad()
        frames = Variable(data[0].float().cuda())
        camparams = Variable(data[1])
        cost, photometric_cost, smoothness_cost, frames, inv_depths, _ = cnnlearner.forward(frames, camparams)
        cost_ = cost.data.cpu()
        inv_depths_mean = inv_depths.mean().data.cpu().numpy()
        cost.backward()
        optimizer.step()

        step_num+=1

        if np.mod(step_num, opt.display_freq)==0:
            frame_vis = frames.data.permute(1,2,0).contiguous().cpu().numpy().astype(np.uint8)
            depth_vis = vis_depthmap(inv_depths.data.cpu()).numpy().astype(np.uint8)
            sio.savemat(os.path.join(opt.checkpoints_dir, 'depth_%s.mat' % (step_num)),{'D': inv_depths.data.cpu().numpy(),'I': frame_vis})

        if np.mod(step_num, opt.save_latest_freq)==0:
            print("cache model....")
            cnnlearner.save_model(os.path.join(opt.checkpoints_dir, '%s' % (epoch)))
            cnnlearner.cuda()
