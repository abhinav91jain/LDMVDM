import torch.nn as nn
from torch.nn import AvgPool2d
from torch.nn.functional import conv2d
from torch.autograd import Variable
import torch
import numpy as np
import os


class ImageSmoothLayer(nn.Module):
    def __init__(self, flag):
        super(ImageSmoothLayer, self).__init__()
        layer = torch.FloatTensor([[0.0751,   0.1238,    0.0751],
                             	  [0.1238,   0.2042,    0.1238],
                            	  [0.0751,   0.1238,    0.0751]]).view(1, 1, 3, 3)
        self.register_buffer('smooth_kernel', layer)
        if flag>1:
            layer1 = layer
            layer = torch.zeros(flag, flag, 3, 3)
            for i in range(flag):
                layer[i, i, :, :] = layer1
        self.register_buffer('smooth_kernel_K', layer)
        self.reflection_pad_func = torch.nn.ReflectionPad2d(1)

    def forward(self, input):
        output_dim = input.dim()
        output_size = input.size()
        if output_dim==2:
            layer = self.smooth_kernel
            input = input.unsqueeze(0).unsqueeze(0)
        elif output_dim==3:
            layer= self.smooth_kernel
            input = input.unsqueeze(1)
        else:
            layer = self.smooth_kernel_K

        x = self.reflection_pad_func(input)
        x = conv2d(input=x,weight=Variable(layer),stride=1, padding=0)
        if output_dim==2:
            x =  x.squeeze(0).squeeze(0)
        elif output_dim==3:
            x =  x.squeeze(1)

        return x


class ImagePyramidLayer(nn.Module):
    def __init__(self, flag, pyramid_layer_num):
        super(ImagePyramidLayer, self).__init__()
        self.pyramid_layer_num = pyramid_layer_num
        layer = torch.FloatTensor([[0.0751,   0.1238,    0.0751],
                              [0.1238,   0.2042,    0.1238],
                              [0.0751,   0.1238,    0.0751]]).view(1, 1, 3, 3)
        self.register_buffer('smooth_kernel', layer)
        if flag>1:
            layer1 = layer
            layer = torch.zeros(flag, flag, 3, 3)
            for i in range(flag):
                layer[i, i, :, :] = layer1
        self.register_buffer('smooth_kernel_K', layer)
        self.avg_pool_func = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.reflection_pad_func = torch.nn.ReflectionPad2d(1)


    def downsample(self, input):
        output_dim = input.dim()
        output_size = input.size()
        if output_dim==2:
            layer = self.smooth_kernel
            input = input.unsqueeze(0).unsqueeze(0)
        elif output_dim==3:
            layer = self.smooth_kernel
            input = input.unsqueeze(1)
        else:
            layer = self.smooth_kernel_K

        x = self.reflection_pad_func(input)
        x = conv2d(input=x, weight=Variable(layer), stride=1,padding=0)
        padding = [0, int(np.mod(input.size(-1), 2)), 0, int(np.mod(input.size(-2), 2))]
        x = torch.nn.ReplicationPad2d(padding)(x)
        x = self.avg_pool_func(x)
        
        if output_dim==2:
            x =  x.squeeze(0).squeeze(0)
        elif output_dim==3:
            x =  x.squeeze(1)

        return x


    def forward(self, input, do_detach=True):
        pyramid = [input]
        for i in range(self.pyramid_layer_num-1):
            img_d = self.downsample(pyramid[i])
            if isinstance(img_d, Variable) and do_detach:
                img_d = img_d.detach()
            pyramid.append(img_d)
            assert(np.ceil(pyramid[i].size(-1)/2) == img_d.size(-1))
        return pyramid


    def get_coords(self, imH, imW):
        x_delta= [np.arange(imW)+.5]
        y_delta= [np.arange(imH)+.5]
        for i in range(self.pyramid_layer_num-1):
            x_delta.append(np.arange(2**i, 2**i + ((2**(i+1))*np.ceil(x_delta[i].shape[0]/2)), 2**(i+1)))
            y_delta.append(np.arange(2**i, 2**i + ((2**(i+1))*np.ceil(y_delta[i].shape[0]/2)), 2**(i+1)))

        return x_delta, y_delta



if __name__ == "__main__":
    img = Variable(torch.randn(3, 128, 416))
    n = ImagePyramidLayer(3, 7)
    x_pyramid, y_pyramid = n.get_coords(128, 416)
    pyramid = n.forward(img)

