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

opt = TrainOptions().parse()
imgSize = [opt.imH, opt.imW]

dataset = KittiDataset("train.txt", opt.dataroot, imgSize, 3)
dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.nThreads, pin_memory=True)

gpu_ids = list(range(opt.batchSize))

