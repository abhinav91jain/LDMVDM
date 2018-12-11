from KittiDataset import KittiDataset
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from ImgPyramid import ImagePyramidLayer
import torch.nn as nn
import torch
import numpy as np
from networks import VggDepthEstimator, PoseNet

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
		self.lkvo = nn.DataParallel(finetuneKernel(imgSize, smoothTerm = smoothTerm), device_ids=gpuIds)
		self.refIdx = refIdx
		self.S = S
		self.use_ssim = use_ssim

	def forward(self, frames, camparams, iterNum=10, level=1):
        cost, photoCost, smoothCost, refFrame, refInv = self.lkvo.forward(frames, camparams, self.refIdx, self.S, iterNum=iterNum, use_ssim=self.use_ssim, level=level)
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
		self.pose_net = PoseNet(3)
        self.depth_net = VggDepthEstimator(img_size)
        self.pyramid_func = ImagePyramidLayer(chan=1, pyramid_layer_num=4)
        self.smoothTerm = smoothTerm

    def forward(self, frames, camparams, refIdx, S=.5, do_data_augment=True, use_ssim=True, iterNum=10, level=1):
        assert(frames.size(0) == 1 and frames.dim() == 5)
        frames = frames.squeeze(0)
        camparams = camparams.squeeze(0).data

        if do_data_augment:
            if np.random.rand()>.5:
                frames = self.fliplr_func(frames)
                camparams[2] = self.img_size[1] - camparams[2]

        bundle_size = frames.size(0)
        src_frame_idx = tuple(range(0,refIdx)) + tuple(range(refIdx+1,bundle_size))
        frames_pyramid = self.vo.pyramid_func(frames)
        ref_frame_pyramid = [frame[refIdx, :, :, :] for frame in frames_pyramid]
        src_frames_pyramid = [frame[src_frame_idx, :, :, :] for frame in frames_pyramid]
        self.vo.setCamera(fx=camparams[0], cx=camparams[2], fy=camparams[4], cy=camparams[5])

        inv_depth_pyramid = self.depth_net.forward((frames-127)/127)
        inv_depth_mean_ten = inv_depth_pyramid[0].mean()*0.1

        inv_depth_norm_pyramid = [depth/inv_depth_mean_ten for depth in inv_depth_pyramid]
        inv_depth0_pyramid = self.pyramid_func(inv_depth_norm_pyramid[0], do_detach=False)
        ref_inv_depth_pyramid = [depth[refIdx, :, :] for depth in inv_depth_norm_pyramid]
        ref_inv_depth0_pyramid = [depth[refIdx, :, :] for depth in inv_depth0_pyramid]
        src_inv_depth_pyramid = [depth[src_frame_idx, :, :] for depth in inv_depth_norm_pyramid]
        src_inv_depth0_pyramid = [depth[src_frame_idx, :, :] for depth in inv_depth0_pyramid]

        self.vo.init(ref_frame_pyramid=ref_frame_pyramid, inv_depth_pyramid=ref_inv_depth0_pyramid)
        p = self.pose_net.forward((frames.view(1, -1, frames.size(2), frames.size(3))-127) / 127)
        rot_mat_batch = self.vo.twist2mat_batch_func(p[0,:,0:3]).contiguous()
        trans_batch = p[0,:,3:6].contiguous()
        rot_mat_batch, trans_batch = self.vo.update_with_init_pose(src_frames_pyramid[0:level], max_itr_num=iterNum, rot_mat_batch=rot_mat_batch, trans_batch=trans_batch)

        photometric_cost = self.vo.compute_phtometric_loss(self.vo.ref_frame_pyramid, src_frames_pyramid, ref_inv_depth_pyramid, src_inv_depth_pyramid, rot_mat_batch, trans_batch, levels=[0,1,2,3], use_ssim=use_ssim)
        smoothness_cost = self.vo.multi_scale_image_aware_smoothness_cost(inv_depth0_pyramid, frames_pyramid, levels=[2,3], type=self.smooth_term) \
                            + self.vo.multi_scale_image_aware_smoothness_cost(inv_depth_norm_pyramid, frames_pyramid, levels=[2,3], type=self.smooth_term)

        cost = photometric_cost + S*smoothness_cost
        return cost, photometric_cost, smoothness_cost, self.vo.ref_frame_pyramid[0], ref_inv_depth0_pyramid[0]*inv_depth_mean_ten

if __name__ = "__main__":
	dataset = KittiDataset()
	dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2, pin_memory=True)
	fineLearner = finetuneLearner(gpu_ids = [0])

	def weights_init(m):
        classname = m.__class__.__name__
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            m.bias.data = torch.zeros(m.bias.data.size())

    fineLearner.apply(weights_init)
    fineLearner.cuda()

    optimizer = optim.Adam(fineLearner.parameters(), lr=.0001)
    for ii, data in enumerate(dataloader):
        optimizer.zero_grad()
        frames = Variable(data[0].float().cuda())
        camparams = Variable(data[1])
        a = fineLearner.forward(frames, camparams)
