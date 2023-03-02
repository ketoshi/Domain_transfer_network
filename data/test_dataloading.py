import argparse
import sys,os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import tensor_to_saveable_img

from dataset import domain_transfer_dataset, RandomCrop, ToTensor, ApplyMask, LightingMult, RotateMult, NormalizeMult

root_dir = '/home/isac/data/viton_hd' 
csv_file = os.path.join(root_dir,'dataset.csv')

trms = transforms.Compose([RandomCrop((512,384)),ToTensor(), \
    ApplyMask(dilate_sz=10,segmentation_to_mask=True), LightingMult(), RotateMult() , NormalizeMult()]) 

dataset = domain_transfer_dataset(csv_file, root_dir, transform=trms)
entry = dataset[0]
dataloader = DataLoader(dataset, batch_size=2,
                        shuffle=True, num_workers=0)

photo = entry['photo']
segmentation = entry['segmentation']
target = entry['target']
plt.imsave('data/test_results/test_image0.jpg', tensor_to_saveable_img(photo))
plt.imsave('data/test_results/test_image1.jpg', tensor_to_saveable_img(segmentation))
plt.imsave('data/test_results/test_image2.jpg', tensor_to_saveable_img(target))

print('complete, check out images.')
