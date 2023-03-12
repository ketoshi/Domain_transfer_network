import argparse
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from utils import tensor_to_saveable_img
from dataset import domain_transfer_dataset, RandomCrop, ToTensor, ApplyMask, ErodeSegmentation, LightingMult, RotateMult, NormalizeMult, Rescale

root_dir = '/home/isac/data/viton_hd' 

trms = torchvision.transforms.Compose([Rescale((512,284)) ,RandomCrop((512,284)), ToTensor(), \
    ApplyMask(dilate_sz=10,segmentation_to_mask=True), ErodeSegmentation(0),LightingMult(), RotateMult() , NormalizeMult()]) 

dataset = domain_transfer_dataset(root_dir, transform=trms)
train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-2,2], generator=torch.Generator().manual_seed(0))
dataset_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=4) 

for ix, x in enumerate(dataset_loader):
    photo = x['photo']
    segmentation = x['segmentation']
    target = x['target'] 
    photo = torchvision.utils.make_grid(photo)
    segmentation = torchvision.utils.make_grid(segmentation)
    target = torchvision.utils.make_grid(target)

plt.imsave('data/test_results/test_image0.jpg', tensor_to_saveable_img(photo))
plt.imsave('data/test_results/test_image1.jpg', tensor_to_saveable_img(segmentation))
plt.imsave('data/test_results/test_image2.jpg', tensor_to_saveable_img(target))

print('complete, check out images.')
