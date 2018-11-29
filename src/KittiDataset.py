from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
from PIL import Image
import os

class KittiDataset(Dataset):
	def __init__(self,listFile,dataPath,imgSize,bundleSize):
		self.dataPath = dataPath
		self.imgSize = imgSize
		self.bundleSize = bundleSize
		self.framePathes = []

		listFile = os.path.join()

	def __len__(self):
		return len(self.framePathes)

	def __getitem__(self,item):

		return frames, camParams

if __name__ == "__main__":
	dataset = KittiDataset()
	dataset.__getitem__(0)
	for data in dataset:
		print(data[1])