import argparse
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from utils import tensor_to_saveable_img, get_dataloader
from dataset import domain_transfer_dataset, RandomCrop, ToTensor, ApplyMask, ErodeSegmentation, LightingMult, RotateMult, NormalizeMult, Rescale, Rescale_bg_down_and_up
import warnings
warnings.filterwarnings('ignore')

root_dir = '/home/isac/data/viton_hd' 

dataset_loader = get_dataloader(root_dir=root_dir,
                            usage='use_model_use_mask',
                            bg_mode="validation",
                            validation_length=3,
                            BATCH_SIZE=3,
                            dilation=0,
                            erosion=0,
                            get_color_segmentation=False
)

trms = torchvision.transforms.Compose([RandomCrop((512,384)), Rescale_bg_down_and_up(), ApplyMask(0), ErodeSegmentation(0), ToTensor(), NormalizeMult() ]) 
dataset = domain_transfer_dataset(root_dir, transform=trms, background_mode="255")
train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-500,500], generator=torch.Generator().manual_seed(0))
dataset_loader = torch.utils.data.DataLoader(val_set, batch_size=3, shuffle=True, num_workers=4)


for x in dataset_loader:
    photo = (x['photo']+1)/2
    segmentation = (x['segmentation']+1)/2
    target = x['target'] 
    a1 = torchvision.utils.make_grid(photo)
    a2 = torchvision.utils.make_grid(target)
    a3 = torchvision.utils.make_grid(segmentation)
    break
grid_tot = torch.concat((a1,a2,a3),dim=1)
plt.imsave('data/test_results/test_image0.jpg', tensor_to_saveable_img(grid_tot))

print('complete, check out images.')
