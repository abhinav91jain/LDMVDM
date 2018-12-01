from KittiDataset import KittiDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import tensorflow as tf
import itertools
from timeit import default_timer as timer


def preprocess_image(self, image):
    # Assuming input image is a single unit
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	return image * 2. -1.


if __name__ = "__main__":
	dataset = KittiDataset()
	loader = DataLoader(dataset,
                            batch_size = 3,
                            img_height = 256,
                            img_width = 256,
                            num_workers = 2)
	with tf.name_scope("data_loading"):
            tgt_image, src_image_stack, intrinsics = loader.load_train_batch()
            tgt_image = self.preprocess_image(tgt_image)
            src_image_stack = self.preprocess_image(src_image_stack)



