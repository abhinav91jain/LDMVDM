from KittiDataset import KittiDataset
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from ImgPyramid import ImagePyramidLayer
import torch.nn as nn
import torch
import numpy as np

class FlipLR(nn.Module):
    def __init__(self, imW, dimW):
        super(FlipLR, self).__init__()
        invIndices = torch.arange(imW-1, -1, -1).long()
        self.register_buffer('invIndices', invIndices)
        self.dimW = dimW

    def forward(self, input):
        return input.index_select(self.dimW, Variable(self.invIndices))

class finetuneLearner(nn.Module):
	def __init__(self, imgSize=[128,416], refIdx=1, S=.5, use_ssim=True, smoothTerm="lap", gpuIds=[0]):
		super(finetuneLearner, self).__init__()
		self.lkvo = nn.DataParallel(finetuneKernel(img_size, smooth_term = smoothTerm), device_ids=gpuIds)
		self.refIdx = refIdx
		self.S = S
		self.use_ssim = use_ssim

	def forward(self, frames, camparams, iterNum=10, level=1):
        cost, photoCost, smoothCost, refFrame, refInv = self.lkvo.forward(frames, camparams, self.refIdx, self.S, max_lk_iter_num=iterNum, use_ssim=self.use_ssim, lk_level=level)
        return cost.mean(), photoCost.mean(), smoothCost.mean(), refFrame, refInv

    def save_model(self, filePath):
        torch.save(self.cpu().lkvo.module.depth_net.state_dict(), filePath)
        self.cuda()

    def load_model(self, depth_net_file_path, pose_net_file_path):
        self.lkvo.module.depth_net.load_state_dict(torch.load(depth_net_file_path))
        self.lkvo.module.pose_net.load_state_dict(torch.load(pose_net_file_path))

    def init_weights(self):
        self.lkvo.module.depth_net.init_weights()

    def get_parameters(self):
        return self.lkvo.module.depth_net.parameters()


class finetuneKernel(nn.Module):

	def __init__(self, imgSize=[128,416], smoothTerm="lap"):
		super(finetuneKernel, self).__init__()
		self.imgSize = imgSize
		self.fliplr_func = FlipLR(imW=img_size[1], dimW=3)
		self.vo = DirectVO(imH=imgSize[0], imW=imgSize[1], pyramid_layer_num=4)
		


if __name__ = "__main__":
	dataset = KittiDataset()
	dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2, pin_memory=True)
