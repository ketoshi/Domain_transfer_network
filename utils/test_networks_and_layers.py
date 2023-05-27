import os
import sys
import torch
import numpy as np
from torchinfo import summary
sys.path.append('utils')
from utils import get_dataloader, tensor_to_saveable_img
from layers import DTCNN, SDTCNN,VGG_SDTCNN
from dataset import RandomCrop, domain_transfer_dataset, AddDilatedBackground, ToErodedMask, A_Norm, A_transforms
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from skimage import io

IS_TEST_MODEL = False

ll=20
LAYER_CHANNELS = (3,ll,ll*2,ll*4,ll*8,ll*16)
BATCH_SIZE = 2
SKIP_LAYERS = [0,1,2,3,4]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_dir = '/home/isac/data/viton_hd'
model = DTCNN(layer_channels=LAYER_CHANNELS, skip_layers=SKIP_LAYERS).to(device)
dataloader = get_dataloader(root_dir=root_dir,
                                usage='train',
                                validation_length=500,
                                BATCH_SIZE=BATCH_SIZE,
                                bg_dilation=0,
                                mask_erosion=0,
)

for ix, x in enumerate(dataloader):
    photo = x['photo'].to(device)
    segmentation = x['segmentation'].to(device)
    if IS_TEST_MODEL:
        output = model(photo,segmentation)
        summary(model, input_size=[photo.shape,segmentation.shape])
    break
print('complete')

