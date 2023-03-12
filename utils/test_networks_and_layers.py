import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchinfo import summary

sys.path.append('data')
from dataset import *
from utils import *
from layers import *

LAYER_CHANNELS = (3,32,64,128,256)
BATCH_SIZE = 3
SKIP_LAYERS = [0,1,2]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_dir = '/home/isac/data/viton_hd'
trms = transforms.Compose([RandomCrop((512,384)), ApplyMask(), \
      ToTensor(), LightingMult(), RotateMult(),  NormalizeMult() ]) 

dataset = domain_transfer_dataset(root_dir, transform=trms)
dataset_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


model = DTCNN(layer_channels=LAYER_CHANNELS, skip_layers=SKIP_LAYERS).to(device)

for ix, x in enumerate(dataset_loader):
    photo = x['photo'].to(device)
    segmentation = x['segmentation'].to(device)
    output = model(photo,segmentation)
    summary(model, input_size=[photo.shape,segmentation.shape])
    break