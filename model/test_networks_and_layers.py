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
from dataset import *
from utils import *
from layers import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_dir = '/home/isac/data/viton_hd'
csv_file = os.path.join(root_dir,'dataset.csv')

trms = transforms.Compose([RandomCrop((512,384)), ApplyMask(), \
      ToTensor(), LightingMult(), RotateMult(),  NormalizeMult() ]) #   , normalize_mult()  

dataset = domain_transfer_dataset(csv_file, root_dir, transform=trms)
dataset_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)#use 4-8 workers, number needs to be tuned so schedule gpu/cpu collaborate (gpu should be faster to proccess than to load a new batch)

print('started')
model = DTCNN(layer_channels=(3,64,128,256,512), skip_layers=[99]).to(device)
for ix, x in enumerate(dataset_loader):
    photo = x['photo'].to(device)
    segmentation = x['segmentation'].to(device)
    output = model(photo,segmentation)
    summary(model, input_size=[photo.shape,segmentation.shape])
    break
'''

    plt.imsave('data/test_results/test_image0.jpg', tensor_to_saveable_img(x['photo'][0,:,:,:]))
    plt.imsave('data/test_results/test_image1.jpg', tensor_to_saveable_img(x['segmentation'][0,:,:,:]))
    plt.imsave('data/test_results/test_image2.jpg', tensor_to_saveable_img(x['target'][0,:,:,:]))
    break

'''
print("program complete, images saved in test_results")