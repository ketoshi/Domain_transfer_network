import sys
import torch
from torch import nn
from torchinfo import summary
from dataset import *

#------------Functions----------------------
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


#---------------MODEL ------------------


class Encoder(nn.Module):
    def __init__(self, layer_channels=(3,64,128,256,512), skip_layers=[-1]):
        super().__init__()
        inputs = []
        for i in range(0, len(layer_channels)-1):
            l1 = layer_channels[i]
            l2 = layer_channels[i+1]
            inputs.append( downscale_layer(l1,l2,False) )
            inputs.append( downscale_layer(l2,l2,True) )
        self.model = nn.Sequential(*inputs)
        self.skip_layers = [2*x+1 for x in skip_layers]

    def forward(self, x):
        outputs = []
        for i_layer in range(len(self.model)):
            x = self.model[i_layer](x)
            if i_layer in self.skip_layers: 
                outputs.append(x)
        outputs.append(x)#only output is given if skip = [-1]
        return outputs

class Shared_Decoder(nn.Module):
    def __init__(self, layer_channels=(512,256,128,64,3), skip_layers=[-1]):
        super().__init__()
        inputs = []
        layer_channels = list(layer_channels)
        skip_layers = [(len(layer_channels)-2) - x for x in skip_layers]
        
        l1 = layer_channels[0]
        inputs.append( upscale_layer(2*l1,  l1,False) ) #concatenate photo & segmentation
        for i in range(0, len(layer_channels)-1):
            l1 = layer_channels[i]
            l2 = layer_channels[i+1]
            if i in skip_layers:      inputs.append( upscale_layer(l1*2,l1,True) )
            else:                     inputs.append( upscale_layer(l1,  l1,True) )
            inputs.append( upscale_layer(l1,l2,False) )
        
        self.model = nn.Sequential(*inputs)
        self.skip_layers = [2*x+1 for x in skip_layers]

    def forward(self, photo, segmentation):
        z = torch.cat([photo[-1], segmentation[-1]], axis=1)
        i_photo = -2
        for i_layer in range(len(self.model)):
            if i_layer in self.skip_layers: 
                z = torch.cat([photo[i_photo], z], axis=1)
                i_photo -=1
            z = self.model[i_layer](z)
        return z

class DTCNN(nn.Module):
    def __init__(self, layer_channels=(3,64,128,256,512), skip_layers=[-1]):
        super().__init__()
        self.photo_encoder        = Encoder(layer_channels = layer_channels,       skip_layers=skip_layers)
        self.segmentation_encoder = Encoder(layer_channels = layer_channels,       skip_layers=skip_layers)
        self.decoder       = Shared_Decoder(layer_channels = layer_channels[::-1], skip_layers=skip_layers)
    def forward(self, photo, segmentation):
        x = self.photo_encoder(photo)
        y = self.segmentation_encoder(segmentation)
        z = self.decoder(x,y)
        return z

class Single_Decoder(nn.Module):
    def __init__(self, layer_channels=(512,256,128,64,3), skip_layers=[-1]):
        super().__init__()
        inputs = []
        layer_channels = list(layer_channels)
        skip_layers = [(len(layer_channels)-2) - x for x in skip_layers]
        
        l1 = layer_channels[0]
        inputs.append( upscale_layer(l1, l1,False) ) #concatenate photo & segmentation
        for i in range(0, len(layer_channels)-1):
            l1 = layer_channels[i]
            l2 = layer_channels[i+1]
            if i in skip_layers:      inputs.append( upscale_layer(l1*2,l1,True) )
            else:                     inputs.append( upscale_layer(l1,  l1,True) )

            inputs.append( upscale_layer(l1,l2,False) )
        
        self.model = nn.Sequential(*inputs)
        self.skip_layers = [2*x+1 for x in skip_layers]

    def forward(self, photo):
        z = photo[-1]
        i_photo = -2
        for i_layer in range(len(self.model)):
            if i_layer in self.skip_layers: 
                z = torch.cat([photo[i_photo], z], axis=1)
                i_photo -=1
            z = self.model[i_layer](z)
        return z

class SDTCNN(nn.Module):
    def __init__(self, layer_channels=(3,64,128,256,512), skip_layers=[-1]):
        super().__init__()
        layer_channels_enc = tuple([4 if x == 3 else x for x in layer_channels])
        self.encoder        = Encoder(layer_channels = layer_channels_enc,       skip_layers=skip_layers)
        self.decoder        = Single_Decoder(layer_channels = layer_channels[::-1], skip_layers=skip_layers)
    def forward(self, photo, segmentation):
        x = torch.cat([photo, segmentation[:,0:1,:,:]], axis=1)
        y = self.encoder(x)
        z = self.decoder(y)
        return z

