import os
import torch
import torchvision
from skimage import io, transform
import matplotlib.pyplot as plt
from utils.dataset import CropAndPad, Outline, A_Norm
import albumentations as A
from torchvision import transforms
import numpy as np
from utils.utils import print_colored_img,print_np_img, get_dataloader, generate_images, write_csv, combine_csv, compare_bg_images, concat_save_2imgs, get_model, tensor_to_saveable_img
root_dir   = '/home/isac/data/viton_hd'
model_path = "/home/isac/data/tensorboard_info/20230402-163312{'dilation': 100, 'erosion': 0, 'p': 0.25, 'skip_layers': [0, 1, 2, 3], 'layer_channels': (3, 64, 128, 256, 512), 'function': 'train_n_epochs'}/test4.pth"
save_folder = '/home/isac/data/use_model_output'

img1 = "/home/isac/data/zalando/photo/img0.jpg"
img2 = "/home/isac/data/zalando/segmentation/img0.png"
dataloader = get_dataloader(root_dir=root_dir,
                            usage='use_mask',
                            bg_mode="train",
                            validation_length=50000,
                            BATCH_SIZE=1,
                            bg_dilation=100,
                            mask_erosion=0,
                    #TODO:
                    #to improve: 1) lighting (hopefully shado fixes it),  2) (backup plan) HDR imaging-luminence estimation (relighting related) (image->hdr)
                    # get rid of this dataloader, and torch.utils.data.DataLoader(
                    # use only neccesary/ data extraction  (priority on 1 image but else multiple, no batch size should be used)   
                    #A.ISONoise(p=1), this augmentation usefull. look up what ISO does!
                    #to improve: 1) lighting (hopefully shado fixes it),  2) (backup plan) HDR imaging-luminence estimation (relighting related) (image->hdr)
                    # if doesn' t got wll can use GANloss
)

#generate_images(dataloader, model_path, save_folder, max_images=4, photo_mode=4)#photo_mode 0,1,2, max images = -1 for generate all photos
#compare_bg_images(dataloader,save_folder)
#concat_save_2imgs(save_path="/home/isac/data/zalando/output/img5.png" , img1="/home/isac/data/zalando/output/img1.png", img2="/home/isac/data/viton_hd/photo/0.jpg")



#code below have some errors still, and only take in image and not folder
if not os.path.isdir(save_folder): os.mkdir(save_folder) 
ix = len(os.listdir(save_folder))
save_path = os.path.join(save_folder,"img"+str(ix)+".png")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(model_path, device)
model.eval()

input_image = io.imread(img1) # even saving image here causes a black screen to be there
#plt.imsave(save_path,input_image)
#print_np_img(input_image)
#print_colored_img(input_image,scale=16)

segmentation_image = io.imread(img2)[:,:,0:3]

trms = transforms.Compose( [CropAndPad(), A_Norm()] )
input_image = torch.unsqueeze( trms(input_image), 0).to(device)
segmentation_image = torch.unsqueeze( trms(segmentation_image), 0).to(device)

generated_images = model(input_image, segmentation_image)
input_image = (input_image+1)/2 
segmentation_image = (segmentation_image+1)/2 

grid0 = torchvision.utils.make_grid(segmentation_image)
grid1 = torchvision.utils.make_grid(input_image)
grid2 = torchvision.utils.make_grid(generated_images)
grid_tot = torch.concat((grid0,grid1,grid2),dim=1)

#plt.imsave(save_path,tensor_to_saveable_img(grid1))
#plt.imsave(save_path,tensor_to_saveable_img(grid1))
#plt.imsave(save_path,tensor_to_saveable_img(grid2))
#plt.imsave(save_path,tensor_to_saveable_img(grid_tot))

#Outline







