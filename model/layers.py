import os
import sys
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn
import csv
from torchinfo import summary

sys.path.append('data')
from dataloader import *


#------------functions----------------------
def convolution_downscale_factor_2(in_channels, out_channels, kernel_size=3):
    kernel_size = (kernel_size//2)*2+1
    padding =  kernel_size//2
    layer =\
        nn.Conv2d(                  
            in_channels = in_channels,  
            out_channels = out_channels, 
            kernel_size = kernel_size,  
            stride = 2,       
            padding = padding       
    )
    return layer

def convolution_upscale_factor_2(in_channels,out_channels, kernel_size=2):
    kernel_size = (kernel_size//2)*2
    padding =  kernel_size//2-1
    layer =\
        nn.ConvTranspose2d(                  
            in_channels = in_channels,  
            out_channels = out_channels, 
            kernel_size = kernel_size,  
            stride = 2,       
            padding = padding      
    )
    return layer

def convolution_change_channel_size(in_channels,out_channels): 
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

def downscale_layer(in_channels, out_channels, should_downscale=True):
    components = []
    components.append( convolution_change_channel_size(in_channels, out_channels))
    components.append(  nn.BatchNorm2d(out_channels) )
    components.append( nn.ReLU() )
    if should_downscale: components.append( nn.MaxPool2d(2) )
    
    layer  = nn.Sequential(*components)
    return layer

def upscale_layer(in_channels, out_channels, should_upscale = True):
    components = []
    if should_upscale: 
        components.append( convolution_upscale_factor_2(in_channels,out_channels) )
    else: 
        components.append( convolution_change_channel_size(in_channels, out_channels) )
    components.append(  nn.BatchNorm2d(out_channels) )
    components.append( nn.ReLU() )
    layer  = nn.Sequential(*components)
    return layer


#---------------classes------------------
class Encoder(nn.Module):
    def __init__(self, layer_channels=(3,64,128,256,512)):
        super().__init__()
        inputs = []
        for i in range(0, len(layer_channels)-1):
            l1 = layer_channels[i]
            l2 = layer_channels[i+1]
            inputs.append( downscale_layer(l1,l2,False) )
            inputs.append( downscale_layer(l2,l2,True) )
        self.model = nn.Sequential(*inputs)
        #self.output_layers = [] #add for decoder

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

#u-net has conv kernel=1x1   at last step, might be worth to atleast try
class Shared_Decoder(nn.Module):
    def __init__(self, layer_channels=(512,256,128,64,3)):
        super().__init__()
        inputs = []
        layer_channels = list(layer_channels)
        layer_channels[0]*=2 #need to double more layers if i use skip connections
        for i in range(0, len(layer_channels)-1):
            l1 = layer_channels[i]
            l2 = layer_channels[i+1]
            inputs.append( upscale_layer(l1,l2,False) )
            inputs.append( upscale_layer(l2,l2,True) )
        self.model = nn.Sequential(*inputs)
        #self.output_layers = [] #add for decoder

    def forward(self, photo, segmentation):
        z = torch.cat([photo, segmentation], axis=1)
        for layer in self.model:
            z = layer(z)
        return z

class DTCNN(nn.Module):
    def __init__(self, layer_channels=(3,64,128)):
        super().__init__()
        self.photo_encoder = Encoder(layer_channels = layer_channels)
        self.segmentation_encoder = Encoder(layer_channels = layer_channels)
        self.decoder = Shared_Decoder(layer_channels = layer_channels[::-1] )

    def forward(self, photo, segmentation):
        x = self.photo_encoder(photo)
        y = self.segmentation_encoder(segmentation)
        z = self.decoder(x,y)
        return z

#use tensorboard to monitor training 
#look it up and use it!


