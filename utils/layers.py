import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

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

def downscale_layer(in_channels, out_channels, should_downscale=True, img_sz=(-1,-1)):
    components = []
    components.append( convolution_change_channel_size(in_channels, out_channels))
    if img_sz[0] < 0:
            components.append(  nn.BatchNorm2d(out_channels) )
    else:   components.append(  nn.LayerNorm((out_channels,*img_sz)) )
    components.append( nn.ReLU() )
    if should_downscale: components.append( nn.MaxPool2d(2) )
    
    layer  = nn.Sequential(*components)
    return layer

def upscale_layer(in_channels, out_channels, should_upscale = True, img_sz=(-1,-1)):
    components = []
    if should_upscale: 
        components.append( convolution_upscale_factor_2(in_channels,out_channels) )
    else: 
        components.append( convolution_change_channel_size(in_channels, out_channels) )
    if img_sz[0]<0:
            components.append(  nn.BatchNorm2d(out_channels) )
    else:   components.append(  nn.LayerNorm((out_channels,*img_sz)) )
    components.append( nn.ReLU() )
    layer  = nn.Sequential(*components)
    return layer

#---------------MODEL ------------------


class Encoder(nn.Module):
    def __init__(self, layer_channels=(3,64,128,256,512), skip_layers=[-1], img_sz=(-1,-1)):
        super().__init__()
        inputs = []
        for i in range(0, len(layer_channels)-1):
            l1 = layer_channels[i]
            l2 = layer_channels[i+1]
            inputs.append( downscale_layer(l1,l2,False,img_sz=img_sz) )
            inputs.append( downscale_layer(l2,l2,True, img_sz=img_sz) )
            img_sz = [im//2 for im in img_sz]
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
    def __init__(self, layer_channels=(512,256,128,64,3), skip_layers=[-1], img_sz=[-1,-1]):
        super().__init__()
        inputs = []
        layer_channels = list(layer_channels)
        skip_layers = [(len(layer_channels)-2) - x for x in skip_layers]
        
        l1 = layer_channels[0]
        inputs.append( upscale_layer(2*l1, l1, False, img_sz=img_sz) ) #concatenate photo & segmentation
        for i in range(0, len(layer_channels)-1):
            l1 = layer_channels[i]
            l2 = layer_channels[i+1]
            img_sz = [2*im for im in img_sz]
            if i in skip_layers:      inputs.append( upscale_layer(l1*2,l1,True, img_sz=img_sz) )
            else:                     inputs.append( upscale_layer(l1,  l1,True, img_sz=img_sz) )
            inputs.append( upscale_layer(l1,l2,False, img_sz=img_sz) )
        
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
    def __init__(self, layer_channels=(3,64,128,256,512), skip_layers=[-1], img_sz=[-1000,-1000]):
        super().__init__()
        img_sz_decoder = [im//(2**(len(layer_channels)-1)) for im in img_sz]
        self.photo_encoder        = Encoder(layer_channels = layer_channels,       skip_layers=skip_layers, img_sz=img_sz)
        self.segmentation_encoder = Encoder(layer_channels = layer_channels,       skip_layers=skip_layers, img_sz=img_sz)
        self.decoder       = Shared_Decoder(layer_channels = layer_channels[::-1], skip_layers=skip_layers, img_sz=img_sz_decoder)
    def forward(self, photo, segmentation):
        x = self.photo_encoder(photo)
        y = self.segmentation_encoder(segmentation)
        z = self.decoder(x,y)
        return z

class Single_Decoder(nn.Module):
    def __init__(self, layer_channels=(512,256,128,64,3), skip_layers=[-1], img_sz=(512,384)):
        super().__init__()
        inputs = []
        layer_channels = list(layer_channels)
        skip_layers = [(len(layer_channels)-2) - x for x in skip_layers]
        
        l1 = layer_channels[0]
        inputs.append( upscale_layer(l1, l1, False, img_sz=img_sz) ) #concatenate photo & segmentation
        for i in range(0, len(layer_channels)-1):
            l1 = layer_channels[i]
            l2 = layer_channels[i+1]
            img_sz = [2*im for im in img_sz]
            if i in skip_layers:      inputs.append( upscale_layer(l1*2,l1, True, img_sz=img_sz) )
            else:                     inputs.append( upscale_layer(l1,  l1, True, img_sz=img_sz) )

            inputs.append( upscale_layer(l1, l2, False, img_sz=img_sz) )
        
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
    def __init__(self, layer_channels=(3,64,128,256,512), skip_layers=[-1], img_sz=(-1000,-1000)):
        super().__init__()
        layer_channels_enc = tuple([4 if x == 3 else x for x in layer_channels])
        self.encoder        = Encoder(layer_channels = layer_channels_enc,       skip_layers=skip_layers, img_sz=img_sz)
        sc = 2**(len(layer_channels)-1)
        img_sz_decode = [im//sc for im in img_sz]
        self.decoder        = Single_Decoder(layer_channels = layer_channels[::-1], skip_layers=skip_layers, img_sz=img_sz_decode)
    def forward(self, photo, segmentation):
        x = torch.cat([photo, segmentation[:,0:1,:,:]], axis=1)
        y = self.encoder(x)
        z = self.decoder(y)
        return z

class GAN_discriminator(nn.Module):
    def __init__(self, layer_channels=(3,64,128,256,512), image_sz=(512,384)):
        super().__init__()
        self.encoder = Encoder(layer_channels = layer_channels, skip_layers=[-1], img_sz=(-1000,-1000))
        exp = 2**(len(layer_channels)-1)
        kernel_sz = (image_sz[0]//exp, image_sz[1]//exp)
        #or potentially have fully connected layer in the end
        self.predict = nn.Conv2d(in_channels=layer_channels[-1], out_channels=1, kernel_size=kernel_sz)
        classifier = [
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=kernel_sz),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ]
        self.classifier = nn.Sequential(*classifier)

    def forward(self, img):
        img = (img-0.5)*2 # normalize, was added after trying gan_with extendd 2 dataset
        x = self.encoder(img)[0]
        x = self.predict(x)# use classifier
        x = torch.sigmoid(x)
        x = torch.round(x)
        return x

#with vggg
class vgg_encoder(nn.Module):
    def __init__(self, layer_channels=(3,64),lay=28):
        super().__init__()
        blocks = []
        if layer_channels[0]==4:
            block1 = [nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)]
            block1.append( vgg19(weights=VGG19_Weights.DEFAULT).features[1:5] )
            block1 = nn.Sequential(*block1)
            blocks.append(block1)
        else: 
            blocks.append(vgg19(weights=VGG19_Weights.DEFAULT).features[0:5] )
        blocks.append(vgg19(weights=VGG19_Weights.DEFAULT).features[5:10] )
        blocks.append(vgg19(weights=VGG19_Weights.DEFAULT).features[10:19])
        if lay != 28: lay=35
        blocks.append(vgg19(weights=VGG19_Weights.DEFAULT).features[19:lay])#28,or 35
        #blocks.append(vgg19(weights=VGG19_Weights.DEFAULT).features[28:37])
        blocks2 = torch.nn.ModuleList(blocks)
        #'''
        for i, layer in enumerate(blocks2):
            if i == 0:  
                if layer_channels[0]==3:
                    for param in layer.parameters():
                        param.requires_grad = True
            else:
                for param in layer.parameters():
                    param.requires_grad = False
        #'''
        self.blocks = blocks2
    def forward(self, x):
        outputs=[]
        for i, block in enumerate(self.blocks):
            x = block(x)
            outputs.append(x)
        #if feature_layers[0]<0:
        outputs.append(x)
        return outputs

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize, device):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(vgg19(weights=VGG19_Weights.DEFAULT).features[:4].eval())#get layer 3 output
        blocks.append(vgg19(weights=VGG19_Weights.DEFAULT).features[4:9].eval())
        blocks.append(vgg19(weights=VGG19_Weights.DEFAULT).features[9:14].eval())
        blocks.append(vgg19(weights=VGG19_Weights.DEFAULT).features[14:18].eval())
        blocks.append(vgg19(weights=VGG19_Weights.DEFAULT).features[18:25].eval())
 
        self.blocks = torch.nn.ModuleList(blocks).to(device)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))
        self.resize = resize
    def forward(self, input, target, feature_layers=[0, 1, 2, 3,4]):
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:# maybe not neccesary since width/height multiple of 32
            input = F.interpolate(input, size=(224,224), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(224,224), mode='bilinear', align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
        return loss

class VGG_SDTCNN(nn.Module):
    def __init__(self, layer_channels=(3,64,128,256,512), skip_layers=[-1], img_sz=(-1000,-1000),lay=28):
        super().__init__()
        layer_channels = (3,64,128,256,512)
        skip_layers = [0,1,2,3,4]

        sc = 2**(len(layer_channels)-1)
        img_sz_decode = [im//sc for im in img_sz]
        self.encoder = vgg_encoder([4],lay=lay)
        self.decoder = Single_Decoder(layer_channels = layer_channels[::-1], skip_layers=skip_layers, img_sz=img_sz_decode)
    
    def forward(self, photo, segmentation):
        x = torch.cat([photo, segmentation[:,0:1,:,:]], axis=1)
        y = self.encoder(x)
        z = self.decoder(y)
        return z

class VGG_discriminator(nn.Module):
    def __init__(self,layer_channels=(3,64), image_sz=(512,384),lay=28):
        super().__init__()
        exp = 2**4
        kernel_sz = (image_sz[0]//exp, image_sz[1]//exp)
        self.encoder = vgg_encoder([3],lay=lay)
        classifier = [
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=kernel_sz),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ]
        self.classifier = nn.Sequential(*classifier)

    def forward(self, img):
        img = (img-0.5)*2 # normalize, was added after trying gan_with extendd 2 dataset
        x = self.encoder(img)[-1]
        x = self.classifier(x)
        x = torch.round(x)
        return x








