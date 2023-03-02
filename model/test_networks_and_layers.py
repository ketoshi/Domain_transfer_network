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
      ToTensor(), LightingMult(), RotateMult(),  NormalizeMult() ]) 

dataset = domain_transfer_dataset(csv_file, root_dir, transform=trms)
dataset_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

model = DTCNN(layer_channels=(3,64,128,256,512), skip_layers=[0,1,2,3]).to(device)
for ix, x in enumerate(dataset_loader):
    photo = x['photo'].to(device)
    segmentation = x['segmentation'].to(device)
    output = model(photo,segmentation)
    summary(model, input_size=[photo.shape,segmentation.shape])
    break