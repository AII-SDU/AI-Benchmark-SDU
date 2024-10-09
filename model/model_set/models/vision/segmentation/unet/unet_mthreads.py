'''
Copyright (c) 2024, 山东大学智能创新研究院(Academy of Intelligent Innovation)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
'''
# Copyright (c) Academy of Intelligent Innovation.
# License-Identifier: BSD 2-Clause License
# AI Benchmark SDU Team

"""
---
title: U-Net
summary: >
    PyTorch implementation and tutorial of U-Net model.
---

# U-Net

This is an implementation of the U-Net model from the paper,
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

U-Net consists of a contracting path and an expansive path.
The contracting path is a series of convolutional layers and pooling layers,
where the resolution of the feature map gets progressively reduced.
Expansive path is a series of up-sampling layers and convolutional layers
where the resolution of the feature map gets progressively increased.

At every step in the expansive path the corresponding feature map from the contracting path
concatenated with the current feature map.

![U-Net diagram from paper](unet.png)

Here is the [training code](experiment.html) for an experiment that trains a U-Net
on [Carvana dataset](carvana.html).
"""

import torch_musa
import torch
import torchvision.transforms.functional
from torch import nn
from model_set.model.model_base import BaseModel
from thop import profile

class DoubleConvolution(nn.Module):
    """
    ### Two $3 \times 3$ Convolution Layers

    Each step in the contraction path and expansive path have two $3 \times 3$
    convolutional layers followed by ReLU activations.

    In the U-Net paper they used $0$ padding,
    but we use $1$ padding so that final feature map is not cropped.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: is the number of input channels
        :param out_channels: is the number of output channels
        """
        super().__init__()

        # First $3 \times 3$ convolutional layer
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        # Second $3 \times 3$ convolutional layer
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # Apply the two convolution layers and activations
        x = self.first(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)


class DownSample(nn.Module):
    """
    ### Down-sample

    Each step in the contracting path down-samples the feature map with
    a $2 \times 2$ max pooling layer.
    """

    def __init__(self):
        super().__init__()
        # Max pooling layer
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


class UpSample(nn.Module):
    """
    ### Up-sample

    Each step in the expansive path up-samples the feature map with
    a $2 \times 2$ up-convolution.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Up-convolution
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)


class CropAndConcat(nn.Module):
    """
    ### Crop and Concatenate the feature map

    At every step in the expansive path the corresponding feature map from the contracting path
    concatenated with the current feature map.
    """
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        """
        :param x: current feature map in the expansive path
        :param contracting_x: corresponding feature map from the contracting path
        """

        # Crop the feature map from the contracting path to the size of the current feature map
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        # Concatenate the feature maps
        x = torch.cat([x, contracting_x], dim=1)
        #
        return x


class unet(nn.Module):
    """
    ## U-Net
    """
    def __init__(self, in_channels=3, out_channels=19):
        """
        :param in_channels: number of channels in the input image
        :param out_channels: number of channels in the result feature map
        """
        super().__init__()

        # Double convolution layers for the contracting path.
        # The number of features gets doubled at each step starting from $64$.
        self.down_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                        [(in_channels, 64), (64, 128), (128, 256), (256, 512)]])
        # Down sampling layers for the contracting path
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])

        # The two convolution layers at the lowest resolution (the bottom of the U).
        self.middle_conv = DoubleConvolution(512, 1024)

        # Up sampling layers for the expansive path.
        # The number of features is halved with up-sampling.
        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in
                                        [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        # Double convolution layers for the expansive path.
        # Their input is the concatenation of the current feature map and the feature map from the
        # contracting path. Therefore, the number of input features is double the number of features
        # from up-sampling.
        self.up_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                      [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        # Crop and concatenate layers for the expansive path.
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])
        # Final $1 \times 1$ convolution layer to produce the output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: input image
        """
        # To collect the outputs of contracting path for later concatenation with the expansive path.
        pass_through = []
        # Contracting path
        for i in range(len(self.down_conv)):
            # Two $3 \times 3$ convolutional layers
            x = self.down_conv[i](x)
            # Collect the output
            pass_through.append(x)
            # Down-sample
            x = self.down_sample[i](x)

        # Two $3 \times 3$ convolutional layers at the bottom of the U-Net
        x = self.middle_conv(x)

        # Expansive path
        for i in range(len(self.up_conv)):
            # Up-sample
            x = self.up_sample[i](x)
            # Concatenate the output of the contracting path
            x = self.concat[i](x, pass_through.pop())
            # Two $3 \times 3$ convolutional layers
            x = self.up_conv[i](x)

        # Final $1 \times 1$ convolution layer
        x = self.final_conv(x)

        #
        return x
    

class unet_mthreads(BaseModel):
    def __init__(self):
        self.input_shape =(1, 3, 640, 640)
        self.device = torch.device('musa' if torch.musa.is_available() else 'cpu')

    def get_input(self):
        self.input = torch.randn(self.input_shape).to(torch.float32).to(self.device)

    def load_model(self):
        self.model = unet().to(self.device)

    def get_params_flops(self) -> list:
        # 'float [params, flops]'
        flops, params = profile(self.model, inputs=(self.input,), verbose=False)
        print("flops, params:",flops, params)
        return [flops, params]

    def inference(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.input)
        return output
    
if __name__ == "__main__":
    model = unet_mthreads()
    model.get_input()
    model.load_model()
    params_flops = model.get_params_flops() 
    import time
    iterations = 100
    for _ in range(10):
        with torch.no_grad():
            image = model.inference()
    t_start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            image = model.inference()

    elapsed_time = time.time() - t_start
    latency = elapsed_time / iterations * 1000
    FPS = 1000 / latency
    print(f"FPS: {FPS:.2f}")