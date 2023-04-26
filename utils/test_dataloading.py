import matplotlib.pyplot as plt
import torch
import torchvision
from utils import tensor_to_saveable_img, get_dataloader
import warnings
warnings.filterwarnings('ignore')

root_dir = '/home/isac/data/viton_hd' 

#v1
dataset_loader = get_dataloader(root_dir=root_dir,
                        usage="train", 
                        bg_mode="train", 
                        validation_length=500, 
                        BATCH_SIZE=3, 
                        bg_dilation=0, 
                        mask_erosion=0, 
                        special_img = "no"
)

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
print('images saved')
