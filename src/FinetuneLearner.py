from KittiDataset import KittiDataset
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable


if __name__ = "__main__":
	dataset = KittiDataset()
	dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2, pin_memory=True)
	