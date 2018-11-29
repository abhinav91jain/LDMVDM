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

		listFile = os.path.join(dataPath,listFile)
		with open(listFile) as file:
			for line in file:
				seqPath, frameName = line.strip().split(" ")
				framePath = os.path.join(seqPath,frameName)
				self.framePathes.append(framePath)

	def __len__(self):
		return len(self.framePathes)

	def __getitem__(self,item):
		camFile = os.path.join(self.dataPath, self.framePathes[item] + "_cam.txt")
		with open(camFile) as file:
            camIntrinsics = [float(x) for x in next(file).split(',')]

        camParams = np.asarray(camIntrinsics)

        imgFile = os.path.join(self.dataPath, self.framePathes[item] + ".jpg")
        framesCat = np.asarray(Image.open(imgFile))

        frameList = []
        for i in range(self.bundleSize):
        	frameList.append(framesCat[:,i*self.imgSize[1]:(i+1)*self.imgSize[1],:])

        frames = np.asarray(frameList).astype(float).transpose(0,3,1,2)
        
		return frames, camParams

if __name__ == "__main__":
	dataset = KittiDataset()
	dataset.__getitem__(0)
	for data in dataset:
		print(data[1])