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

dataset = KittiDataset("train.txt", "", [128, 416], 3)
dataloader = DataLoader(dataset, batch_size="", shuffle=True, num_workers="", pin_memory=True)
